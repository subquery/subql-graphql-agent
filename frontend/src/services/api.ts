import axios from 'axios';
import type {
  Project,
  ProjectConfig,
  RegisterProjectRequest,
  RegisterProjectResponse,
  UpdateProjectConfigRequest,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ProjectsListResponse,
  HealthResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

console.log('API Base URL:', API_BASE_URL);

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url, config.data);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url, response.data);
    return response;
  },
  (error) => {
    console.error('API Response Error:', {
      status: error.response?.status,
      url: error.config?.url,
      data: error.response?.data,
      message: error.message,
    });
    return Promise.reject(error);
  }
);

export const projectsApi = {
  // Register a new project
  register: async (request: RegisterProjectRequest): Promise<RegisterProjectResponse> => {
    const response = await api.post<RegisterProjectResponse>('/register', request);
    return response.data;
  },

  // List all projects
  list: async (): Promise<ProjectsListResponse> => {
    const response = await api.get<ProjectsListResponse>('/projects');
    return response.data;
  },

  // Get project configuration
  getConfig: async (cid: string): Promise<ProjectConfig> => {
    const response = await api.get<ProjectConfig>(`/projects/${cid}`);
    return response.data;
  },

  // Update project configuration
  updateConfig: async (cid: string, updates: UpdateProjectConfigRequest): Promise<ProjectConfig> => {
    const response = await api.patch<ProjectConfig>(`/projects/${cid}`, updates);
    return response.data;
  },

  // Delete project
  delete: async (cid: string): Promise<{ cid: string; deleted: boolean; message: string }> => {
    const response = await api.delete(`/projects/${cid}`);
    return response.data;
  },

  // Chat with project (non-streaming)
  chat: async (cid: string, request: ChatCompletionRequest): Promise<ChatCompletionResponse> => {
    const response = await api.post<ChatCompletionResponse>(`/${cid}/chat/completions`, request);
    return response.data;
  },

  // Health check
  health: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },
};

export const chatApi = {
  // Stream chat responses
  streamChat: async function* (
    cid: string, 
    request: ChatCompletionRequest
  ): AsyncGenerator<string, void, unknown> {
    const url = `${API_BASE_URL}/${cid}/chat/completions`;
    console.log('Streaming chat to:', url);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Failed to get response reader');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            if (data === '[DONE]') {
              return;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.choices?.[0]?.delta?.content) {
                yield parsed.choices[0].delta.content;
              }
            } catch (e) {
              console.warn('Failed to parse SSE data:', data);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },
};