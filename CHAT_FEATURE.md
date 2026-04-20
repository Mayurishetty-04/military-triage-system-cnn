# 💬 Live Chat Feature - Implementation Guide

## Overview
A real-time messaging system between doctors and patients has been successfully implemented into the Military Triage System. This feature enables instant communication during triage assessment.

---

## 🎯 What's New

### Backend Changes
1. **New Database Models** (`backend/app/models.py`):
   - `Message` - Stores individual chat messages
   - `Conversation` - Manages chat sessions between patient and doctor

2. **New API Endpoints** (`backend/app/main.py`):
   - `POST /messages/send` - Send a message to a recipient
   - `GET /messages/conversations` - Get all conversations for current user
   - `GET /messages/conversation/{id}` - Fetch messages from specific conversation

### Frontend Changes
1. **New Chat Component** (`frontend/src/components/ChatPanel.jsx`):
   - Beautiful chat interface with slide-in animation
   - Real-time message sending and receiving
   - Auto-scroll to latest messages
   - Mark messages as read functionality

2. **Chat Styling** (`frontend/src/components/ChatPanel.css`):
   - Gradient-based dark theme matching system design
   - Responsive mobile-friendly layout
   - Smooth animations and transitions

3. **Integration**:
   - Chat button added to Triage page header
   - ChatPanel component integrated in Triage flow
   - Conversation management and history retrieval

---

## 🚀 How to Use

### For Patients
1. Go to the Triage page
2. Click the **💬 Chat** button in the top-right corner
3. Start typing your message and click 📤 to send
4. Chat history is automatically loaded and displayed
5. Messages are marked as read when you view them

### For Doctors
1. Go to the Doctor Dashboard
2. Access the chat panel (same button)
3. Select a patient conversation
4. Send and receive messages in real-time

---

## 🔌 API Endpoints

### Send Message
```
POST /messages/send
Authorization: Bearer <token>

Body:
{
  "content": "Hello doctor",
  "recipient_id": 2
}

Response:
{
  "id": 1,
  "content": "Hello doctor",
  "sender_id": 1,
  "sender_username": "patient1",
  "timestamp": "2026-04-20T10:30:00",
  "is_read": 0
}
```

### Get All Conversations
```
GET /messages/conversations
Authorization: Bearer <token>

Response:
[
  {
    "id": 1,
    "patient_id": 1,
    "patient_username": "patient1",
    "doctor_id": 2,
    "doctor_username": "doctor1",
    "last_message": "Your vitals look good",
    "updated_at": "2026-04-20T10:30:00",
    "unread_count": 0
  }
]
```

### Get Conversation Messages
```
GET /messages/conversation/{conversation_id}
Authorization: Bearer <token>

Response:
[
  {
    "id": 1,
    "content": "Hello doctor",
    "sender_id": 1,
    "sender_username": "patient1",
    "timestamp": "2026-04-20T10:25:00",
    "is_read": 1
  }
]
```

---

## 💾 Database Schema

### Message Table
```sql
CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER FOREIGN KEY,
  sender_id INTEGER FOREIGN KEY,
  content VARCHAR,
  timestamp DATETIME DEFAULT now,
  is_read INTEGER DEFAULT 0
)
```

### Conversation Table
```sql
CREATE TABLE conversations (
  id INTEGER PRIMARY KEY,
  patient_id INTEGER FOREIGN KEY,
  doctor_id INTEGER FOREIGN KEY,
  created_at DATETIME DEFAULT now,
  updated_at DATETIME DEFAULT now,
  UNIQUE(patient_id, doctor_id)
)
```

---

## ✨ Features

✅ **Real-time Messaging** - Send and receive messages instantly  
✅ **Conversation History** - All messages are persisted in database  
✅ **Read Status** - Track which messages have been read  
✅ **Auto-Create Conversations** - Conversations are auto-created on first message  
✅ **User-Friendly UI** - Beautiful gradient-based dark theme  
✅ **Mobile Responsive** - Works on desktop, tablet, and mobile  
✅ **Message Timestamps** - Every message shows when it was sent  
✅ **Authorization** - Only authenticated users can access chat  

---

## 🔐 Security

- All endpoints require authentication (`Bearer token`)
- Users can only view conversations they're part of
- Messages are stored securely in SQLite database
- JWT-based authorization for all routes

---

## 🚀 Quick Start

1. **Backend is Ready** - Database and API endpoints are fully functional
2. **Frontend is Ready** - Chat component is integrated into Triage page
3. **Test It**:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload
   ```
4. **Open Frontend** - Navigate to Triage page and click Chat button

---

## 📊 Future Enhancements

- 🔔 **Push Notifications** - Alert users of new messages
- ⏱️ **Typing Indicators** - Show when someone is typing
- 🖼️ **Image Sharing** - Send medical images in chat
- 📝 **Message Search** - Find past conversations
- 🔊 **Voice Messages** - Send audio instead of text
- 👥 **Group Chat** - Multiple doctors + patient conversations
- ⚡ **WebSocket Support** - True real-time updates without polling

---

## 📝 Notes

- Messages are currently fetched via HTTP polling (not WebSocket)
- For production, consider implementing WebSocket for true real-time updates
- Database stores all messages for audit trail and compliance
- Chat panel is modal-based and can be opened/closed freely

---

Created: April 20, 2026  
Status: ✅ Complete and Tested
