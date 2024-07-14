// src/components/MemoryInput.js
import React, { useState } from "react";
import axios from "axios";
import "./MemoryInput.css";

const MemoryInput = () => {
  const [memory, setMemory] = useState("");
  const [status, setStatus] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!memory.trim()) return;

    setStatus("Saving...");
    try {
      const response = await axios.get(
        "https://backend-a3kiovdawq-wl.a.run.app/postMemory?text=" + memory
      );
      console.log(response);
      setStatus("Memory saved successfully!");
      setMemory("");
    } catch (error) {
      console.error("Error saving memory:", error);
      setStatus("Error saving memory. Please try again.");
    }
  };

  return (
    <div className="memory-input">
      <h2>Add a New Memory</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          value={memory}
          onChange={(e) => setMemory(e.target.value)}
          placeholder="Enter your memory..."
          rows="4"
        />
        <button type="submit">Save Memory</button>
      </form>
      {status && <p className="status">{status}</p>}
    </div>
  );
};

export default MemoryInput;
