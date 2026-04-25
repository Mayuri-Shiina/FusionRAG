const { createApp } = Vue;

createApp({
  data() {
    return {
      token: localStorage.getItem("fusionrag_token") || "",
      currentUser: null,
      authMode: "login",
      authLoading: false,
      authForm: { username: "", password: "", role: "user", admin_code: "" },
      view: "chat",
      selectedFile: null,
      parseMode: "auto",
      uploading: false,
      uploadJob: {},
      uploadPollTimer: null,
      documents: [],
      deletingDocuments: {},
      sessions: [],
      openingSessionId: "",
      sessionId: `session_${Date.now()}`,
      messages: [],
      userInput: "",
      loading: false
    };
  },
  computed: {
    isAuthenticated() {
      return Boolean(this.token && this.currentUser);
    },
    isAdmin() {
      return this.currentUser?.role === "admin";
    }
  },
  async mounted() {
    if (this.token) {
      try {
        await this.fetchMe();
      } catch (_) {
        this.logout();
      }
    }
  },
  beforeUnmount() {
    this.stopUploadPolling();
  },
  methods: {
    authHeaders(extra = {}) {
      return this.token ? { ...extra, Authorization: `Bearer ${this.token}` } : extra;
    },
    async authFetch(url, options = {}) {
      const response = await fetch(url, {
        ...options,
        headers: this.authHeaders(options.headers || {})
      });
      if (response.status === 401) {
        this.logout();
        throw new Error("登录已过期，请重新登录");
      }
      return response;
    },
    async fetchMe() {
      const response = await this.authFetch("/auth/me");
      if (!response.ok) throw new Error("认证失败");
      this.currentUser = await response.json();
    },
    async submitAuth() {
      const username = this.authForm.username.trim();
      const password = this.authForm.password.trim();
      if (!username || !password) {
        alert("用户名和密码不能为空");
        return;
      }
      this.authLoading = true;
      try {
        const endpoint = this.authMode === "login" ? "/auth/login" : "/auth/register";
        const payload = { username, password };
        if (this.authMode === "register") {
          payload.role = this.authForm.role;
          payload.admin_code = this.authForm.admin_code || null;
        }
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "认证失败");
        this.token = data.access_token;
        localStorage.setItem("fusionrag_token", this.token);
        this.currentUser = { username: data.username, role: data.role };
        this.authForm.password = "";
        this.authForm.admin_code = "";
        this.view = "chat";
      } catch (error) {
        alert(error.message);
      } finally {
        this.authLoading = false;
      }
    },
    logout() {
      this.token = "";
      this.currentUser = null;
      this.messages = [];
      this.sessions = [];
      this.documents = [];
      localStorage.removeItem("fusionrag_token");
    },
    showChat() {
      this.view = "chat";
    },
    async showDocuments() {
      this.view = "documents";
      await this.loadDocuments();
    },
    async showSessions() {
      this.view = "sessions";
      await this.loadSessions();
    },
    onFileSelected(event) {
      this.selectedFile = event.target.files?.[0] || null;
      this.uploadJob = {};
    },
    async uploadDocument() {
      if (!this.selectedFile) return;
      this.uploading = true;
      this.uploadJob = { message: "正在上传文件", status: "uploading", done_list: [], running_list: [] };
      try {
        const formData = new FormData();
        formData.append("file", this.selectedFile);
        formData.append("parse_mode", this.parseMode);
        const response = await this.authFetch("/documents/upload", {
          method: "POST",
          body: formData
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "上传失败");
        this.uploadJob = { ...data, status: "processing", done_list: [], running_list: [] };
        this.startUploadPolling(data.task_id || data.job_id);
      } catch (error) {
        this.uploadJob = { message: error.message, status: "failed", done_list: [], running_list: [] };
        this.uploading = false;
      }
    },
    startUploadPolling(taskId) {
      this.stopUploadPolling();
      const poll = async () => {
        try {
          const response = await this.authFetch(`/documents/status/${encodeURIComponent(taskId)}`);
          const data = await response.json();
          this.uploadJob = { ...this.uploadJob, ...data };
          if (data.status === "completed" || data.status === "failed") {
            this.uploading = false;
            this.stopUploadPolling();
            await this.loadDocuments();
          }
        } catch (error) {
          this.uploadJob = { ...this.uploadJob, message: error.message, status: "failed" };
          this.uploading = false;
          this.stopUploadPolling();
        }
      };
      poll();
      this.uploadPollTimer = setInterval(poll, 1200);
    },
    stopUploadPolling() {
      if (this.uploadPollTimer) {
        clearInterval(this.uploadPollTimer);
        this.uploadPollTimer = null;
      }
    },
    async loadDocuments() {
      const response = await this.authFetch("/documents");
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "加载文档失败");
      this.documents = data.documents || [];
    },
    async deleteDocument(filename) {
      if (!confirm(`确定删除 ${filename} 的向量数据吗？`)) return;
      const previous = [...this.documents];
      this.deletingDocuments = { ...this.deletingDocuments, [filename]: true };
      this.documents = this.documents.filter((doc) => doc.filename !== filename);
      try {
        const response = await this.authFetch(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "删除失败");
        setTimeout(() => this.loadDocuments().catch(() => {}), 600);
      } catch (error) {
        this.documents = previous;
        alert(error.message);
      } finally {
        const next = { ...this.deletingDocuments };
        delete next[filename];
        this.deletingDocuments = next;
      }
    },
    async loadSessions() {
      const response = await this.authFetch("/sessions");
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "加载会话失败");
      this.sessions = data.sessions || [];
    },
    async loadSession(sessionId) {
      if (!sessionId || this.openingSessionId === sessionId) return;
      this.openingSessionId = sessionId;
      try {
        const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`);
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "加载会话失败");
        this.sessionId = sessionId;
        this.messages = (data.messages || []).map((item) => ({
          text: item.content,
          isUser: item.type === "human",
          imageUrls: this.cleanImageUrls(item.image_urls || [])
        }));
        this.view = "chat";
        this.scrollToBottom();
      } catch (error) {
        alert(error.message || "加载会话失败");
      } finally {
        this.openingSessionId = "";
      }
    },
    async deleteSession(sessionId) {
      if (!confirm(`确定删除会话 ${sessionId} 吗？`)) return;
      await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      await this.loadSessions();
      if (this.sessionId === sessionId) this.newChat();
    },
    newChat() {
      this.sessionId = `session_${Date.now()}`;
      this.messages = [];
      this.view = "chat";
    },
    handleKeydown(event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.sendMessage();
      }
    },
    async sendMessage() {
      const text = this.userInput.trim();
      if (!text || this.loading) return;
      this.messages.push({ text, isUser: true });
      this.userInput = "";
      this.loading = true;
      this.messages.push({ text: "", isUser: false, steps: [], imageUrls: [] });
      const botIdx = this.messages.length - 1;
      this.scrollToBottom();

      try {
        const response = await this.authFetch("/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: this.sessionId })
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let eventEnd;
          while ((eventEnd = buffer.indexOf("\n\n")) !== -1) {
            const eventText = buffer.slice(0, eventEnd);
            buffer = buffer.slice(eventEnd + 2);
            if (!eventText.startsWith("data: ")) continue;
            const payload = eventText.slice(6);
            if (payload === "[DONE]") continue;
            const data = JSON.parse(payload);
            if (data.type === "content") {
              this.messages[botIdx].text += data.content || "";
            } else if (data.type === "rag_step") {
              this.messages[botIdx].steps.push(data.step);
            } else if (data.type === "final") {
              const final = data.content || {};
              if (final.answer) this.messages[botIdx].text = this.stripImageBlock(final.answer);
              this.messages[botIdx].imageUrls = this.cleanImageUrls(final.image_urls || []);
            } else if (data.type === "error") {
              this.messages[botIdx].text += `\n\n[错误] ${data.content}`;
            }
          }
          this.scrollToBottom();
        }
      } catch (error) {
        this.messages[botIdx].text = `请求失败：${error.message}`;
      } finally {
        this.loading = false;
        this.scrollToBottom();
      }
    },
    parseMarkdown(text) {
      return marked.parse(text || "");
    },
    stripImageBlock(text) {
      return (text || "")
        .replace(/【图片】[\s\S]*$/g, "")
        .replace(/!\[[^\]]*]\((?:https?:\/\/)?(?:www\.)?example\.com\/[^)]*\)/gi, "")
        .replace(/https?:\/\/(?:www\.)?example\.com\/\S+/gi, "")
        .trim();
    },
    formatAssistantText(text) {
      return marked.parse(this.stripImageBlock(text));
    },
    cleanImageUrls(urls) {
      const seen = new Set();
      return (urls || [])
        .map((url) => String(url || "").trim())
        .filter((url) => url && !/example\.com|image_placeholder/i.test(url))
        .filter((url) => {
          if (seen.has(url)) return false;
          seen.add(url);
          return true;
        });
    },
    imageSrc(url) {
      return `/assets/image?url=${encodeURIComponent(url)}`;
    },
    imageName(url) {
      try {
        const pathname = new URL(url).pathname;
        return decodeURIComponent(pathname.split("/").filter(Boolean).pop() || "相关图片");
      } catch (_) {
        return "相关图片";
      }
    },
    escapeHtml(text) {
      const div = document.createElement("div");
      div.textContent = text || "";
      return div.innerHTML;
    },
    scrollToBottom() {
      this.$nextTick(() => {
        if (this.$refs.chatScroll) {
          this.$refs.chatScroll.scrollTop = this.$refs.chatScroll.scrollHeight;
        }
      });
    }
  }
}).mount("#app");
