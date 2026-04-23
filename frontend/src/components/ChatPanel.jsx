import React, { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import { BASE_URL } from "../config";
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
        `${BASE_URL}/messages/conversations`,
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
          `${BASE_URL}/messages/conversation/${conv.id}`,
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
        `${BASE_URL}/messages/send`,
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
    <div className={`chat-panel-overlay ${isOpen ? 'open' : ''}`} onClick={onClose}>
      <div className="chat-panel glass-panel border-white/10" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="chat-header border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-cyan-500/10 rounded-xl flex items-center justify-center text-xl">
              📡
            </div>
            <div>
              <h3 className="chat-title font-['Outfit'] font-black tracking-tight">{recipientName || "Field Medic"}</h3>
              <div className="chat-subtitle font-mono text-[8px] uppercase tracking-widest text-cyan-500/60">
                Secure Link: Established
              </div>
            </div>
          </div>
          <button className="w-8 h-8 rounded-full hover:bg-white/5 flex items-center justify-center text-gray-500 transition-colors" onClick={onClose}>
            ✕
          </button>
        </div>

        {/* Messages Container */}
        <div className="chat-messages custom-scrollbar">
          {messages.length === 0 ? (
            <div className="chat-empty font-mono text-[10px] uppercase tracking-widest text-gray-600">
              Initializing Communication History...
            </div>
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
                <div className="message-content font-medium text-sm">{msg.content}</div>
                <div className="message-time font-mono text-[9px] uppercase tracking-tighter opacity-40 mt-1">
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
        <div className="chat-input-area border-t border-white/5">
          <input
            type="text"
            className="chat-input glass-card"
            placeholder="Enter secure message..."
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            disabled={!recipientId}
          />
          <button
            className="chat-send-btn bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 border border-cyan-500/30 transition-all"
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
