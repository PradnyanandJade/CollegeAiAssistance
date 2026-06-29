import { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const [response, setResponse] = useState({
    answer: "",
    source_found: false,
  });

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);

    try {
        const response = await fetch(
        "http://127.0.0.1:8000/ask",
        {
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                question
            })
        }
        );

      const data = await response.json();

      setResponse(data);
    } catch (err) {
      setResponse({
        answer: "Server not running.",
        source_found: false,
      });
    }

    setLoading(false);
  };

  return (
    <div className="container">

      <div className="card">

        <h1>College AI Assistant</h1>

        <p>
          Ask questions about admissions, fees, placements, scholarships,
          hostel, examinations and more.
        </p>
        <br></br>
        <div className="inputRow">

          <input
            type="text"
            placeholder="Ask your question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />

          <button onClick={askQuestion}>
            Ask
          </button>

        </div>

        {loading && (
          <div className="loading">
            Thinking...
          </div>
        )}

        {response.answer && (

          <div className="result">

            <h2>Answer</h2>

            <p>{response.answer}</p>

            <div
              className={
                response.source_found
                  ? "status success"
                  : "status error"
              }
            >
              {response.source_found
                ? "✅ Information Found"
                : "❌ Information Not Found"}
            </div>

          </div>

        )}

      </div>

    </div>
  );
}

export default App;