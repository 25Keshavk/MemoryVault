import React, { useState, useEffect } from "react";
import axios from "axios";
import "./QueryPage.css";

const QueryPage = () => {
  // ... (keep all the existing state and functions)

  return (
    <div className="memory-vault">
      <div className="sidebar">
        <button onClick={startNewConversation} className="new-conversation-btn">
          New Conversation
        </button>
        <div className="conversation-list">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => selectConversation(conv.id)}
              className={`conversation-btn ${
                currentConversationId === conv.id ? "active" : ""
              }`}
            >
              {new Date(parseInt(conv.id)).toLocaleString()}
            </button>
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
