import React, { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import "./ChatPanel.css";

const ChatPanel = ({ recipientId, recipientName, isOpen, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [currentUserId, setCurrentUserId] = useState(null);
  const messagesEndRef = useRef(null);
  const pollIntervalRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Get current user ID from token
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        const id = payload.user_id ?? payload.userId ?? payload.id ?? null;
        setCurrentUserId(id != null ? Number(id) : null);
      } catch (e) {
        console.error("Failed to decode token:", e);
      }
    }
  }, []);

  const fetchConversation = useCallback(async () => {
    try {
      const token = localStorage.getItem("token");
      const rid = recipientId != null ? Number(recipientId) : null;
      if (!token || !rid || !currentUserId) return;

      // Get conversations to find the one with this recipient
      const convRes = await axios.get(
        "http://127.0.0.1:8000/messages/conversations",
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // Find conversation - works both ways (patient->doctor or doctor->patient)
      const conv = convRes.data.find((c) => {
        const patientId = Number(c.patient_id);
        const doctorId = c.doctor_id != null ? Number(c.doctor_id) : null;
        return (
          (patientId === rid && doctorId === currentUserId) ||
          (doctorId === rid && patientId === currentUserId)
        );
      });

      if (conv) {
        // Fetch messages from this conversation
        const msgRes = await axios.get(
          `http://127.0.0.1:8000/messages/conversation/${conv.id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setMessages(msgRes.data);
      } else {
        // No existing conversation - start fresh
        setMessages([]);
      }
    } catch (err) {
      console.error("Error fetching conversation:", err);
    }
  }, [recipientId, currentUserId]);

  // Fetch conversation and messages
  useEffect(() => {
    const rid = recipientId != null ? Number(recipientId) : null;
    if (isOpen && rid && currentUserId) {
      fetchConversation();
      // Poll for new messages every 2 seconds
      pollIntervalRef.current = setInterval(() => {
        fetchConversation();
      }, 2000);
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [isOpen, recipientId, currentUserId, fetchConversation]);

  const sendMessage = async () => {
    const rid = recipientId != null ? Number(recipientId) : null;
    if (!rid) {
      alert("No chat recipient selected.");
      return;
    }
    if (!newMessage.trim()) return;

    try {
      const token = localStorage.getItem("token");
      const res = await axios.post(
        "http://127.0.0.1:8000/messages/send",
        {
          content: newMessage,
          recipient_id: rid,
        },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // Add message to local state immediately
      setMessages([...messages, res.data]);
      setNewMessage("");

      // Refresh conversation list
      fetchConversation();

      // Show notification
      showNotification(`Message sent to ${recipientName}`);
    } catch (err) {
      console.error("Error sending message:", err);
      alert("Failed to send message");
    }
  };

  const showNotification = (title) => {
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification(title, {
        icon: "💬",
        tag: "chat-notification",
      });
    }
  };

  if (!isOpen) return null;

  return (
    <div className="chat-panel-overlay" onClick={onClose}>
      <div className="chat-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="chat-header">
          <div>
            <h3 className="chat-title">💬 {recipientName || "Chat"}</h3>
            <div className="chat-subtitle">
              You: {currentUserId ?? "—"} · Recipient: {recipientId ?? "—"}
            </div>
          </div>
          <button className="chat-close-btn" onClick={onClose}>
            ✕
          </button>
        </div>

        {/* Messages Container */}
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="chat-empty">Start the conversation...</div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`chat-message ${
                  Number(msg.sender_id) === Number(currentUserId)
                    ? "sent"
                    : "received"
                }`}
              >
                <div className="message-content">{msg.content}</div>
                <div className="message-time">
                  {new Date(msg.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="chat-input-area">
          <input
            type="text"
            className="chat-input"
            placeholder="Type a message..."
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            disabled={!recipientId}
          />
          <button
            className="chat-send-btn"
            onClick={sendMessage}
            disabled={!recipientId || !newMessage.trim()}
          >
            📤
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;
