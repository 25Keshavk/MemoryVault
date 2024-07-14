import React, { useState } from "react";
import axios from "axios";
import "./QueryPage.css";

const QueryPage = () => {
  const [queries, setQueries] = useState([]);
  const [currentQuery, setCurrentQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!currentQuery.trim()) return;

    // Immediately add the query to the list
    const newQuery = { query: currentQuery, response: null, isLoading: true };
    setQueries([...queries, newQuery]);
    setCurrentQuery("");
    setIsLoading(true);

    try {
      const response = await axios.get(
        "https://backend-a3kiovdawq-wl.a.run.app/query?query=" + currentQuery
      );

      console.log(response);

      // Update the query with the response
      setQueries((prevQueries) =>
        prevQueries.map((q, index) =>
          index === prevQueries.length - 1
            ? { ...q, response: response.data, isLoading: false }
            : q
        )
      );
    } catch (error) {
      console.error("Error fetching response:", error);
      // Update the query to show an error
      setQueries((prevQueries) =>
        prevQueries.map((q, index) =>
          index === prevQueries.length - 1
            ? { ...q, response: "Error fetching response", isLoading: false }
            : q
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="memory-vault">
      <div className="conversation">
        {queries.map((item, index) => (
          <div key={index} className="query-response">
            <p className="query">{item.query}</p>
            {item.isLoading ? (
              <div className="loading-animation">Loading...</div>
            ) : (
              <p className="response">{item.response}</p>
            )}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="query-form">
        <input
          type="text"
          value={currentQuery}
          onChange={(e) => setCurrentQuery(e.target.value)}
          placeholder="Enter your query..."
          className="query-input"
          disabled={isLoading}
        />
        <button type="submit" className="submit-button" disabled={isLoading}>
          Send
        </button>
      </form>
    </div>
  );
};

export default QueryPage;
