let threadId = localStorage.getItem("threadId") || crypto.randomUUID();
let token = localStorage.getItem("accessToken") || "";
let databaseEnabled = false;
let registerMode = false;

const authView = document.querySelector("#auth-view");
const appView = document.querySelector("#app-view");
const authForm = document.querySelector("#auth-form");
const authError = document.querySelector("#auth-error");
const authTitle = document.querySelector("#auth-title");
const authCopy = document.querySelector("#auth-copy");
const authSubmit = document.querySelector("#auth-submit");
const authToggle = document.querySelector("#auth-toggle");
const chat = document.querySelector("#chat");
const welcome = document.querySelector("#welcome");
const form = document.querySelector("#form");
const messageInput = document.querySelector("#message");
const fileInput = document.querySelector("#files");
const fileList = document.querySelector("#file-list");
const threads = document.querySelector("#threads");
const sidebar = document.querySelector("#sidebar");
const healthStatus = document.querySelector(".status-dot");

function authHeaders() {
  return token ? {Authorization: `Bearer ${token}`} : {};
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {...authHeaders(), ...(options.headers || {})},
  });
  if (response.status === 204) return null;
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Request failed");
  return result;
}

async function checkHealth() {
  try {
    await request("/health");
    healthStatus.textContent = "Online";
    healthStatus.classList.remove("offline");
  } catch {
    healthStatus.textContent = "Unavailable";
    healthStatus.classList.add("offline");
  }
}

function showAuth() {
  authView.classList.remove("hidden");
  appView.classList.add("hidden");
}

function showApp(user) {
  authView.classList.add("hidden");
  appView.classList.remove("hidden");
  const email = user?.email || "Local user";
  document.querySelector("#user-email").textContent = email;
  document.querySelector("#user-name").textContent = user?.display_name || email.split("@")[0];
  document.querySelector("#user-avatar").textContent = email[0].toUpperCase();
}

function setAuthMode(isRegister) {
  registerMode = isRegister;
  authTitle.textContent = isRegister ? "Create your account" : "Welcome back";
  authCopy.textContent = isRegister
    ? "Start saving your conversations and files."
    : "Sign in to continue your conversations.";
  authSubmit.textContent = isRegister ? "Create account" : "Sign in";
  authToggle.textContent = isRegister
    ? "Already have an account? Sign in"
    : "New here? Create an account";
  authError.textContent = "";
}

function emptyChat() {
  chat.replaceChildren();
  const section = document.createElement("section");
  section.id = "welcome";
  section.className = "welcome";
  section.innerHTML = '<div class="brand-mark">M</div><h2>How can I help you today?</h2><p>Ask a question or upload a PDF, image, or audio file.</p>';
  chat.append(section);
  document.querySelector("#thread-title").textContent = "New conversation";
}

function addMessage(text, role) {
  document.querySelector("#welcome")?.remove();
  const row = document.createElement("article");
  row.className = `message-row ${role}`;
  const content = document.createElement("div");
  content.className = "message-content";
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "You" : "M";
  const body = document.createElement("div");
  body.className = "message-body";
  if (role === "assistant") {
    body.innerHTML = DOMPurify.sanitize(marked.parse(text || ""));
  } else {
    body.textContent = text;
  }
  content.append(avatar, body);
  row.append(content);
  chat.append(row);
  chat.scrollTop = chat.scrollHeight;
  return body;
}

function addDetails(target, title, text) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  const content = document.createElement("div");
  summary.textContent = title;
  content.className = "meta";
  content.textContent = text;
  details.append(summary, content);
  target.append(details);
}

async function loadThreads() {
  threads.replaceChildren();
  if (!databaseEnabled || !token) return [];
  const items = await request("/threads");
  items.forEach((item) => {
    const row = document.createElement("button");
    row.className = `thread${item.id === threadId ? " active" : ""}`;
    row.dataset.threadId = item.id;
    const title = document.createElement("span");
    title.className = "thread-title";
    title.textContent = item.title;
    const remove = document.createElement("span");
    remove.className = "thread-delete";
    remove.textContent = "�";
    remove.title = "Delete conversation";
    remove.addEventListener("click", async (event) => {
      event.stopPropagation();
      await request(`/threads/${item.id}`, {method: "DELETE"});
      if (threadId === item.id) {
        threadId = crypto.randomUUID();
        localStorage.setItem("threadId", threadId);
        emptyChat();
      }
      await loadThreads();
    });
    row.append(title, remove);
    row.addEventListener("click", () => openThread(item.id));
    threads.append(row);
  });
  return items;
}

async function openThread(id) {
  const thread = await request(`/threads/${id}`);
  threadId = id;
  localStorage.setItem("threadId", id);
  threads.querySelectorAll(".thread").forEach((row) => {
    row.classList.toggle("active", row.dataset.threadId === id);
  });
  document.querySelector("#thread-title").textContent = thread.title;
  chat.replaceChildren();
  thread.messages.forEach((message) => {
    if (message.role === "user" || message.role === "assistant") {
      addMessage(message.content, message.role);
    }
  });
  if (!thread.messages.length) emptyChat();
  sidebar.classList.remove("open");
}

async function startNewThread() {
  emptyChat();
  if (databaseEnabled && token) {
    const thread = await request("/threads", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({title: "New Conversation"}),
    });
    threadId = thread.id;
  } else {
    threadId = crypto.randomUUID();
  }
  localStorage.setItem("threadId", threadId);
  sidebar.classList.remove("open");
  await loadThreads();
  messageInput.focus();
}

async function checkAuth() {
  try {
    const result = await request("/auth/me");
    databaseEnabled = result.database_enabled;
    if (databaseEnabled && !result.user) return showAuth();
    showApp(result.user);
    const items = await loadThreads();
    const current = items.find((item) => item.id === threadId);
    if (current) await openThread(current.id);
    else emptyChat();
  } catch {
    token = "";
    localStorage.removeItem("accessToken");
    showAuth();
  }
}

authToggle.addEventListener("click", () => setAuthMode(!registerMode));
authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authError.textContent = "";
  authSubmit.disabled = true;
  try {
    const result = await request(registerMode ? "/auth/register" : "/auth/login", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        email: document.querySelector("#email").value.trim(),
        password: document.querySelector("#password").value,
      }),
    });
    token = result.access_token;
    localStorage.setItem("accessToken", token);
    threadId = crypto.randomUUID();
    localStorage.setItem("threadId", threadId);
    await checkAuth();
  } catch (error) {
    authError.textContent = error.message;
  } finally {
    authSubmit.disabled = false;
  }
});

document.querySelector("#logout").addEventListener("click", () => {
  token = "";
  localStorage.removeItem("accessToken");
  setAuthMode(false);
  showAuth();
});
document.querySelector("#new-thread").addEventListener("click", startNewThread);
document.querySelector("#open-sidebar").addEventListener("click", () => sidebar.classList.add("open"));
document.querySelector("#close-sidebar").addEventListener("click", () => sidebar.classList.remove("open"));

fileInput.addEventListener("change", () => {
  fileList.textContent = [...fileInput.files].map((file) => file.name).join(" � ");
});
messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 180)}px`;
});
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  const files = [...fileInput.files];
  if (!message && !files.length) return;
  const data = new FormData();
  data.append("message", message);
  data.append("thread_id", threadId);
  files.forEach((file) => data.append("files", file));
  addMessage(message || `Uploaded: ${files.map((file) => file.name).join(", ")}`, "user");
  messageInput.value = "";
  messageInput.style.height = "auto";
  fileInput.value = "";
  fileList.textContent = "";
  const pending = addMessage("Working...", "assistant");
  document.querySelector("#send").disabled = true;

  try {
    const result = await request("/chat", {method: "POST", body: data});
    threadId = result.thread_id;
    localStorage.setItem("threadId", threadId);
    pending.innerHTML = DOMPurify.sanitize(marked.parse(result.answer || ""));
    if (result.extracted) addDetails(pending, "Extracted content", result.extracted);
    addDetails(pending, "How this answer was produced", (result.logs || []).join("\n"));
    await loadThreads();
    const active = [...threads.querySelectorAll(".thread")].find((item) => item.classList.contains("active"));
    document.querySelector("#thread-title").textContent = active?.querySelector(".thread-title")?.textContent || message.slice(0, 60) || "New conversation";
  } catch (error) {
    pending.textContent = error.message;
    pending.classList.add("error");
  } finally {
    document.querySelector("#send").disabled = false;
  }
});

setAuthMode(false);
checkHealth();
checkAuth();
