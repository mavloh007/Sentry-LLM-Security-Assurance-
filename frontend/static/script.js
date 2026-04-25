const messagesContainer = document.getElementById('messagesContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const historySidebar = document.getElementById('historySidebar');
const historyList = document.getElementById('historyList');
const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
const newSessionBtn = document.getElementById('newSessionBtn');

let conversations = [];
let activeConversationId = '';
let pendingTitle = '';

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

toggleSidebarBtn.addEventListener('click', () => {
    historySidebar.classList.toggle('hidden');
});

newSessionBtn.addEventListener('click', () => {
    activeConversationId = '';
    pendingTitle = `New chat · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    clearMessages();
    renderWelcomeMessage();
    renderHistoryList();
    userInput.focus();
});

async function boot() {
    await refreshConversations();

    if (activeConversationId) {
        await loadConversation(activeConversationId);
    } else {
        renderWelcomeMessage();
    }
}

async function refreshConversations() {
    try {
        const response = await fetch('/api/conversations', {
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        conversations = Array.isArray(data.conversations) ? data.conversations : [];
        activeConversationId = data.active_conversation_id || activeConversationId || '';
        renderHistoryList();
    } catch (error) {
        console.error('Failed to load conversations:', error);
        conversations = [];
        renderHistoryList();
    }
}

function renderHistoryList() {
    historyList.innerHTML = '';

    const draftBtn = document.createElement('button');
    draftBtn.type = 'button';
    draftBtn.className = `history-item${activeConversationId ? '' : ' active'}`;
    draftBtn.innerHTML = `
        <div class="history-title">${escapeHtml(pendingTitle || 'Start New Chat')}</div>
        <div class="history-meta">Draft</div>
    `;
    draftBtn.addEventListener('click', () => {
        activeConversationId = '';
        clearMessages();
        renderWelcomeMessage();
        renderHistoryList();
    });
    historyList.appendChild(draftBtn);

    conversations.forEach((conversation, index) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `history-item${conversation.id === activeConversationId ? ' active' : ''}`;

        const label = formatConversationLabel(conversation, index + 1);
        const meta = formatConversationMeta(conversation);

        btn.innerHTML = `
            <div class="history-title">${escapeHtml(label)}</div>
            <div class="history-meta">${escapeHtml(meta)}</div>
        `;

        btn.addEventListener('click', async () => {
            await selectConversation(conversation.id);
        });

        historyList.appendChild(btn);
    });
}

function formatConversationLabel(conversation, fallbackNumber) {
    const baseTitle = (conversation.title || `Conversation ${fallbackNumber}`).trim();
    return baseTitle || `Conversation ${fallbackNumber}`;
}

function formatConversationMeta(conversation) {
    let createdText = '';
    if (conversation.created_at) {
        const dt = new Date(conversation.created_at);
        if (!Number.isNaN(dt.getTime())) {
            createdText = dt.toLocaleString([], {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        }
    }

    const shortId = (conversation.id || '').replace(/-/g, '').slice(-4);
    const parts = [];
    if (createdText) parts.push(createdText);
    if (shortId) parts.push(`#${shortId}`);
    return parts.join(' · ') || 'Conversation';
}

async function selectConversation(conversationId) {
    if (!conversationId || conversationId === activeConversationId) return;

    try {
        const response = await fetch(`/api/conversations/${conversationId}/select`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        activeConversationId = conversationId;
        pendingTitle = '';
        renderHistoryList();
        await loadConversation(conversationId);
    } catch (error) {
        console.error('Failed to switch conversation:', error);
    }
}

async function loadConversation(conversationId) {
    if (!conversationId) {
        clearMessages();
        renderWelcomeMessage();
        return;
    }

    try {
        const response = await fetch(`/api/conversations/${conversationId}/messages`, {
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        clearMessages();

        const messages = Array.isArray(data.messages) ? data.messages : [];
        if (messages.length === 0) {
            renderWelcomeMessage();
        } else {
            messages.forEach((message) => {
                addMessage(message.content || '', message.role || 'assistant');
            });
        }

        scrollToBottom();
    } catch (error) {
        console.error('Failed to load conversation:', error);
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    userInput.value = '';
    userInput.focus();
    sendBtn.disabled = true;

    removeWelcomeMessage();
    addMessage(message, 'user');
    scrollToBottom();

    try {
        const loadingId = showLoading();

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                conversation_id: activeConversationId,
                conversation_title: activeConversationId ? undefined : deriveTitleFromMessage(message),
                new_session: !activeConversationId
            })
        });

        removeLoading(loadingId);

        if (!response.ok) {
            if (response.status === 401) {
                addMessage('Session expired. Please log in again.', 'error');
                window.location.href = '/login';
                return;
            }
            const maybeJson = await safeReadJson(response);
            const backendError = maybeJson && maybeJson.error ? maybeJson.error : `HTTP error! status: ${response.status}`;
            throw new Error(backendError);
        }

        const data = await response.json();

        if (data.error) {
            addMessage(data.error, 'error');
        } else {
            addMessage(data.response, 'assistant');
            if (data.conversation_id) {
                activeConversationId = data.conversation_id;
                pendingTitle = '';
            }
            // Use the conversation list returned in the chat response
            // instead of making a separate GET /api/conversations call
            if (data.conversations) {
                conversations = Array.isArray(data.conversations) ? data.conversations : [];
                activeConversationId = data.active_conversation_id || activeConversationId;
            }
            renderHistoryList();
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage((error && error.message) || 'Sorry, an error occurred. Please try again.', 'error');
    } finally {
        sendBtn.disabled = false;
        scrollToBottom();
    }
}

function deriveTitleFromMessage(message) {
    const text = (message || '').trim();
    if (!text) return 'New Conversation';
    const words = text.split(/\s+/).slice(0, 8).join(' ');
    return words.length < text.length ? `${words}...` : words;
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
     if (sender === 'assistant' && window.marked && window.DOMPurify) {
        const html = window.marked.parse(text || '', { breaks: true });
        contentDiv.innerHTML = window.DOMPurify.sanitize(html);
    } else {
        // user + error messages stay as plain text — never trust user input as HTML
        contentDiv.textContent = text;
    }
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
}

function renderWelcomeMessage() {
    if (document.getElementById('welcomeMessage')) return;

    const welcomeMessage = document.createElement('div');
    welcomeMessage.className = 'welcome-message';
    welcomeMessage.id = 'welcomeMessage';
    welcomeMessage.innerHTML = `
        <h2>Welcome to SGBank Chatbot</h2>
        <p>Ask me anything about withdrawals, emergency procedures, identity verification, or fraud prevention.</p>
    `;
    messagesContainer.appendChild(welcomeMessage);
}

function removeWelcomeMessage() {
    const welcomeMsg = messagesContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
}

function clearMessages() {
    messagesContainer.innerHTML = '';
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.innerHTML = `
        <div class="loading">
            <span></span><span></span><span></span>
            <span>Assistant is thinking...</span>
        </div>
    `;
    const loadingId = 'loading-' + Date.now();
    loadingDiv.id = loadingId;
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();
    return loadingId;
}

function removeLoading(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function safeReadJson(response) {
    try {
        return await response.json();
    } catch {
        return null;
    }
}

function escapeHtml(text) {
    return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

boot();
