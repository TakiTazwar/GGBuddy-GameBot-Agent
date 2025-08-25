"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]); // history of { prompt, response }

  const sendQuery = async () => {
    if (!prompt.trim()) return;

    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/game-query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Cookie': 'csrftoken=eEW2o57eQPmNCqisMr74GbKIUUNlmpR6'
        },
        body: JSON.stringify({ prompt })
      });

      const data = await res.json();
      const responseText = data.response || JSON.stringify(data);

      setMessages(prev => [...prev, { prompt, response: responseText }]);
      setPrompt('');
    } catch (err) {
      setMessages(prev => [...prev, { prompt, response: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-10 p-4 space-y-4 border rounded shadow">
      <h1 className="text-xl font-bold">GG:Buddy</h1>
      <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-2">
        {messages.map((msg, index) => (
          <div key={index} className="p-3 border rounded bg-white space-y-2">
            <p className="text-black"><strong>You:</strong> {msg.prompt}</p>
            <div className="text-black">
              <strong>Bot:</strong>
              <div className="prose prose-sm">
                <ReactMarkdown>{msg.response}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="space-y-2">
        <input
          type="text"
          value={prompt}
          placeholder="Ask about a game's future player count..."
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full p-2 border rounded"
        />
        <button
          onClick={sendQuery}
          disabled={loading}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          {loading ? 'Asking...' : 'Ask the Bot'}
        </button>
      </div>
    </div>
  );
}
