from fastapi import FastAPI, HTTPException,Request  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import requests
import re
import time
import ast
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager

# === CONFIG ===
hf_model_name = "BAAI/bge-small-en"
index_path = "./faiss_index/index.faiss"
texts_path = "./faiss_index/texts.json"

# === Device + Model Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = AutoModel.from_pretrained(hf_model_name).to(device)
model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Load FAISS index ===
    try:
        app.state.index = faiss.read_index(index_path)
        print("FAISS index loaded successfully.")
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        app.state.index = None

    # === Load Texts Safely ===
    try:
        with open(texts_path, "r", encoding="utf-8") as f:
            app.state.texts = json.load(f)
            print(f"{len(app.state.texts)} texts loaded.")
    except MemoryError:
        print("MemoryError: Too large to load entire texts.json.")
        app.state.texts = []
    except Exception as e:
        print(f"Error loading texts.json: {e}")
        app.state.texts = []

    yield  # App is now ready to serve

    # Optional: Cleanup if needed

# === Initialize FastAPI App ===
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/check-faiss")
def check_status(request: Request):
    index_status = "loaded" if request.app.state.index else "not loaded"
    texts_count = len(request.app.state.texts)
    return {
        "faiss_index": index_status,
        "texts_loaded": texts_count
    }


url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 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",
    "Cookie": "__cf_bm=.1bYeE7jv0WDJd2qcYh9EapT1o9w0TZNhOXfAw2KLdw-1752616809-1.0.1.1-KKPBFz6QZria6VTVNQyHeTVLKj661azYm14F2nX6BHG7TXxaNYiae03nxa1w8gfh._SgBqw.rtkKC_HvkHuXeYRBldb3NVWdIS2oyV8f46s; _cfuvid=kVoVPzb0vFsQQ._84UvAGzO_MO9APWn6nnwRSo3Ajq8-1752596930275-0.0.1.1-604800000"
}

# In-memory prompt history
prompt_history: List[str] = []
prompt_response: List[str] = []

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    response: str

@app.post("/game-query", response_model=PromptResponse)
def handle_prompt(request: PromptRequest):
    
    

    # === Embedding Function ===
    def embed_texts(texts: list[str]) -> np.ndarray:
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).expand(out.size()).float()
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = (summed / counts).cpu().numpy().astype("float32")
        return embeddings

    # === Search Function ===
    def search_faiss(prompt: str, top_k: int = 5) -> list[str]:
        query_vec = embed_texts([prompt])
        D, I = app.state.index.search(query_vec, top_k)
        return [app.state.texts[i] for i in I[0]]

    # === Callable Function ===
    def run_search(prompt: str, top_k: int = 5):
        results = search_faiss(prompt, top_k=top_k)
        return results  # optionally return results too
    
    def callOpenAPI(total_content):
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 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",  # Replace with your real key
            "Cookie": "__cf_bm=.1bYeE7jv0WDJd2qcYh9EapT1o9w0TZNhOXfAw2KLdw-1752616809-1.0.1.1-KKPBFz6QZria6VTVNQyHeTVLKj661azYm14F2nX6BHG7TXxaNYiae03nxa1w8gfh._SgBqw.rtkKC_HvkHuXeYRBldb3NVWdIS2oyV8f46s; _cfuvid=kVoVPzb0vFsQQ._84UvAGzO_MO9APWn6nnwRSo3Ajq8-1752596930275-0.0.1.1-604800000"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "store": True,
            "messages": [
                {"role": "user", "content": total_content}
            ]
        }
        
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["message"]["content"]


    def getReviewCheck(prompt):
        review_content =  f"""Classify if the prompt intend to get user reviews. Only Reply the binary classification [yes,no]:
                                            Prompt: {prompt}
                """
        return callOpenAPI(review_content)
        
    def getDateFormat(prompt):
        date_prompt =  f"""From the prompt give me data in format(numeric) [year,month] otherwise return [none]:
                                            Prompt: {prompt}
                """
        date_value_string = callOpenAPI(date_prompt)
        s = date_value_string.replace("none", "None")
        try:
            date_value = ast.literal_eval(s)
            if isinstance(date_value, list):
                return date_value
            else:
                raise ValueError("Not a list")
        except Exception as e:
            print("⚠️ Parse error:", e)
            return []

    def getAppID(text):
        match = re.search(r'appid:\s*(\d+)', text)
        if match:
            app_id = match.group(1)
            return app_id
        else:
            return None

    def clean_retrieved_entry(entry: str) -> str:
        try:
            data = json.loads(entry)
            return f"{data.get('name', '')}: {data.get('short_description', '')}"
        except Exception:
            return entry[:500]


    def getForeCastingCheck(prompt):
        forecast_prompt =  f"""Classify if the prompt intend to forecast player count of the game. Only Reply the binary classification [yes,no]:
                                            Prompt: {prompt}
                """
        return callOpenAPI(forecast_prompt)
        
    def get_monthly_data(appid: int) -> pd.DataFrame:
        url = f"https://steamcharts.com/app/{appid}/chart-data.json"
        response = requests.get(url)
        raw_data = response.json()

        df = pd.DataFrame(raw_data, columns=["timestamp", "players"])

        # Use correct unit: milliseconds
        df = df[df["timestamp"].between(1100000000000, 1900000000000)]
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Resample to monthly averages
        df = df.set_index("datetime").resample("M").mean(numeric_only=True).reset_index()
        df = df[df["players"].notnull()]
        if len(df) < 2:
            raise ValueError(f"❌ Not enough valid monthly data points: {len(df)}")

        df["month_index"] = np.arange(len(df))
        return df

    def predict_for(df: pd.DataFrame, target_year: int, target_month: int):
        """
        Predict average players for any specified year and month — past or future.
        Uses linear regression on the month index.
        """
        model = LinearRegression()
        X = df[["month_index"]]
        y = df["players"]
        model.fit(X, y)

        # Build target date
        target_date = pd.Timestamp(year=target_year, month=target_month, day=1)

        # Compute month index difference from first date
        first_date = df["datetime"].iloc[0]
        delta_months = (target_date.year - first_date.year) * 12 + (target_date.month - first_date.month)
        future_index = delta_months

        predicted = model.predict([[future_index]])
        return target_date, predicted[0]


    def get_top_reviews(appid, max_reviews=20, fetch_limit=100):
        all_reviews = []
        cursor = "*"

        while len(all_reviews) < fetch_limit:
            url = f"https://store.steampowered.com/appreviews/{appid}"
            params = {
                "json": 1,
                "num_per_page": min(100, fetch_limit - len(all_reviews)),
                "cursor": cursor,  # FIXED: removed quote
                "language": "english",
                "filter": "recent",  # or "updated"
                "review_type": "all"
            }

            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Request failed with status code {response.status_code}")
                break

            data = response.json()

            reviews = data.get("reviews", [])
            if not reviews:
                print("No reviews found.")
                break

            all_reviews.extend(reviews)
            cursor = data.get("cursor", "")
            time.sleep(1.0)

        # Sort reviews by most helpful (votes_up) and get the top max_reviews
        sorted_reviews = sorted(all_reviews, key=lambda r: r["votes_up"], reverse=True)
        top_reviews = [r["review"] for r in sorted_reviews[:max_reviews]]

        return top_reviews

    prompt = request.prompt

    content = ""
    for conv_number, past_prompt in enumerate(prompt_history):
        content += f"""
        You are a game AI Assistant:

        Previous conversation: 
        Old Query: {past_prompt}
        Old Response: {prompt_response[conv_number]}
        """

    try:
        search_data = run_search(prompt)
        cleaned_result = clean_retrieved_entry(search_data[0])
        appid = getAppID(search_data[0])

        # Forecast Handling
        if getForeCastingCheck(prompt).lower() == "yes":
            date_format = getDateFormat(prompt)
            monthly_players = get_monthly_data(appid)

            if len(date_format) == 2:
                target_date, predicted_players = predict_for(
                    monthly_players, target_year=date_format[0], target_month=date_format[1]
                )
                actual_predicted_player = int(predicted_players)
                content += f"""
                Current Operation:
                Use the Data if it is related to prompt.

                Prompt: {prompt}

                Predicted Player Count: {actual_predicted_player}

                Predicted Player Count is already calculated, Now write a message according to the prompt with the Player count given.
                """
            else:
                content += f"""
                Prompt: {prompt}

                Write sorry, Something bad occurred. Try to send the message again. Thank you.
                """
        elif getReviewCheck(prompt).lower() == "yes":
            top_10_reviews = get_top_reviews(appid, max_reviews=10)
            combined = " ".join(top_10_reviews)
            content += f"""
            Current Operation:

            This is the User Reviews of the mentioned Game. Use this Data as User Reviews:
            User Reviews: "{combined}"

            Prompt: "{prompt}"

            Execute the User prompt and provide a clean concise result, return as simple text.
            """
        else:
            content += f"""
            Current Operation:
            Use the Data below if it is related to the user query.
            --- Retrieved Data ---
            {cleaned_result}

            --- User Prompt ---
            {prompt}

            Please generate a clean, helpful answer in plain text.
            """

        response = callOpenAPI(content)

        # Save to history
        prompt_history.append(prompt)
        prompt_response.append(str(response))

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # return {"message": "Hello, FastAPI!"}
