import { useEffect, useRef } from 'react';
import axios from 'axios';
import { BASE_URL } from '../config';

const useMessageNotifications = () => {
  const lastNotifiedByConvIdRef = useRef(new Map());

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) return;

    const checkMessages = async () => {
      try {
        const res = await axios.get(`${BASE_URL}/messages/conversations`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        // Check for new unread messages
        const conversations = res.data;

        for (const conv of conversations) {
          if (conv.unread_count > 0) {
            const senderName = conv.patient_username || conv.doctor_username || 'Someone';
            const updatedAt = conv.updated_at ? new Date(conv.updated_at).getTime() : Date.now();
            const lastNotifiedAt = lastNotifiedByConvIdRef.current.get(conv.id) || 0;

            // Avoid notifying on every poll; only notify when conversation updates.
            if (updatedAt <= lastNotifiedAt) continue;
            lastNotifiedByConvIdRef.current.set(conv.id, updatedAt);
            
            // Show notification
            if ('Notification' in window && Notification.permission === 'granted') {
              const notification = new Notification(`New message from ${senderName}`, {
                icon: '💬',
                tag: `chat-${conv.id}`,
                body: conv.last_message || 'New message',
              });

              // Click to focus window
              notification.onclick = () => {
                window.focus();
                notification.close();
              };
            }
          }
        }
      } catch (err) {
        console.error('Error checking messages:', err);
      }
    };

    // Check for messages every 3 seconds
    const interval = setInterval(checkMessages, 3000);
    // Run once immediately (after mount) to populate lastNotified map and reduce false positives.
    checkMessages();

    return () => clearInterval(interval);
  }, []);
};

export default useMessageNotifications;
