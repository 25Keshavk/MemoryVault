import React, { useState, useEffect } from "react";
import axios from "axios";
import "./QueryPage.css";

const QueryPage = () => {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [queries, setQueries] = useState([]);
  const [currentQuery, setCurrentQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);

  const suggestedPrompts = [
    "Tell me about a happy memory",
    "Tell me about a time I spent with my family",
    "What was my favorite childhood toy?",
    "Describe a memorable vacation I took",
  ];

  useEffect(() => {
    const savedConversations = JSON.parse(
      localStorage.getItem("conversations") || "[]"
    );
    const sortedConversations = savedConversations.sort((a, b) => b.id - a.id);
    setConversations(sortedConversations);

    if (
      sortedConversations.length === 0 ||
      !localStorage.getItem("hasVisited")
    ) {
      startNewConversation();
      localStorage.setItem("hasVisited", "true");
    } else {
      setCurrentConversationId(sortedConversations[0].id);
      setQueries(sortedConversations[0].queries);
    }
  }, []);

  const saveToLocalStorage = (updatedConversations) => {
    localStorage.setItem("conversations", JSON.stringify(updatedConversations));
  };

  const startNewConversation = () => {
    const newId = Date.now().toString();
    const newConversation = { id: newId, queries: [] };
    const updatedConversations = [newConversation, ...conversations];
    setConversations(updatedConversations);
    setCurrentConversationId(newId);
    setQueries([]);
    saveToLocalStorage(updatedConversations);
  };

  const deleteConversation = (id, event) => {
    event.stopPropagation(); // Prevent triggering selectConversation
    const updatedConversations = conversations.filter((conv) => conv.id !== id);
    setConversations(updatedConversations);
    saveToLocalStorage(updatedConversations);

    if (id === currentConversationId) {
      if (updatedConversations.length > 0) {
        setCurrentConversationId(updatedConversations[0].id);
        setQueries(updatedConversations[0].queries);
      } else {
        setCurrentConversationId(null);
        setQueries([]);
      }
    }
  };

  const formatQueryHistory = (queries, newQuery) => {
    let formattedHistory = queries
      .map(
        (item, index) =>
          `query ${index + 1}:\n${item.query}\n\nresponse ${index + 1}:\n${
            item.response
          }\n\n`
      )
      .join("");

    formattedHistory += `new query:\n${newQuery}`;
    return formattedHistory;
  };

  const handlePromptClick = (prompt) => {
    setShowWelcome(false);
    startNewConversation();
    setCurrentQuery(prompt);
    handleSubmit(null, prompt);
  };

  const handleSubmit = async (e, manualPrompt = null) => {
    if (e) e.preventDefault();
    const queryToSend = manualPrompt || currentQuery;
    if (!queryToSend.trim()) return;

    const newQuery = { query: queryToSend, response: null, isLoading: true };
    const updatedQueries = [...queries, newQuery];
    setQueries(updatedQueries);
    setCurrentQuery("");
    setIsLoading(true);

    const formattedQuery = formatQueryHistory(queries, currentQuery);

    try {
      const response = await axios.get("http://127.0.0.1:8080/query", {
        params: { query: formattedQuery },
      });

      console.log(response);

      const updatedQueriesWithResponse = updatedQueries.map((q, index) =>
        index === updatedQueries.length - 1
          ? { ...q, response: response.data, isLoading: false }
          : q
      );

      setQueries(updatedQueriesWithResponse);

      const updatedConversations = conversations.map((conv) =>
        conv.id === currentConversationId
          ? { ...conv, queries: updatedQueriesWithResponse }
          : conv
      );

      setConversations(updatedConversations);
      saveToLocalStorage(updatedConversations);
    } catch (error) {
      console.error("Error fetching response:", error);
      setQueries(
        updatedQueries.map((q, index) =>
          index === updatedQueries.length - 1
            ? { ...q, response: "Error fetching response", isLoading: false }
            : q
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const selectConversation = (id) => {
    setCurrentConversationId(id);
    const selectedConversation = conversations.find((conv) => conv.id === id);
    setQueries(selectedConversation.queries);
  };

  const renderWelcomeOverlay = () => (
    <div className="welcome-overlay">
      <div className="welcome-content">
        <h2>Welcome to MemoryVault</h2>
        <p>Here are some suggested prompts to get started:</p>
        <div className="suggested-prompts">
          {suggestedPrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => handlePromptClick(prompt)}
              className="prompt-btn"
            >
              {prompt}
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowWelcome(false)}
          className="close-welcome-btn"
        >
          Close
        </button>
      </div>
    </div>
  );

  return (
    <div className="memory-vault">
      {showWelcome && renderWelcomeOverlay()}
      <div className="sidebar">
        <button onClick={startNewConversation} className="new-conversation-btn">
          New Conversation
        </button>
        <div className="conversation-list">
          {conversations.map((conv) => (
            <div key={conv.id} className="conversation-item">
              <button
                onClick={() => selectConversation(conv.id)}
                className={`conversation-btn ${
                  currentConversationId === conv.id ? "active" : ""
                }`}
              >
                {new Date(parseInt(conv.id)).toLocaleString()}
              </button>
              <button
                onClick={(e) => deleteConversation(conv.id, e)}
                className="delete-btn"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>
      <div className="main-content">
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
    </div>
  );
};

export default QueryPage;
