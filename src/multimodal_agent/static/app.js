const threadId = crypto.randomUUID();
const chat = document.querySelector("#chat");
const form = document.querySelector("#form");
const messageInput = document.querySelector("#message");
const fileInput = document.querySelector("#files");

function scrollToLatest() {
  scrollTo(0, document.body.scrollHeight);
}

function addMessage(text, className) {
  const message = document.createElement("div");
  message.className = "msg " + className;
  message.textContent = text;
  chat.append(message);
  scrollToLatest();
  return message;
}

function addDetails(title, text) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  const content = document.createElement("div");

  summary.textContent = title;
  content.className = "meta";
  content.textContent = text;
  details.append(summary, content);
  chat.append(details);
}

function renderMarkdown(element, text) {
  element.innerHTML = DOMPurify.sanitize(marked.parse(text || ""));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const files = [...fileInput.files];
  const data = new FormData();
  data.append("thread_id", threadId);
  data.append("message", messageInput.value);
  files.forEach((file) => data.append("files", file));

  const userText = messageInput.value
    || "Uploaded: " + files.map((file) => file.name).join(", ");
  addMessage(userText, "user");

  messageInput.value = "";
  fileInput.value = "";
  const pending = addMessage("Working...", "agent");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      body: data,
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Request failed");
    }

    renderMarkdown(pending, result.answer);

    if (result.extracted) {
      addDetails("Extracted content", result.extracted);
    }

    addDetails("Plan and logs", (result.logs || []).join("\n"));
    scrollToLatest();
  } catch (error) {
    pending.classList.add("error");
    pending.textContent = error.message;
  }
});
