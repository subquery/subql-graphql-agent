import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, MessageCircle, Trash2 } from 'lucide-react';
import { useChat } from '../hooks/useChat';
import { useProject } from '../hooks/useProjects';
import { formatTimestamp, formatCid } from '../lib/utils';
import type { ChatMessage } from '../types';

interface ChatInterfaceProps {
  projectCid: string;
  messages: ChatMessage[];
  onMessagesChange: (messagesOrUpdater: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => void;
  onClearMessages: () => void;
}

// 新增：可折叠的think块组件
function ThinkBlock({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false); // 默认展开
  const contentRef = useRef<HTMLDivElement>(null);
  const userIsScrollingRef = useRef(false);
  const lastContentUpdateRef = useRef(Date.now());
  const autoScrollTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  // 检测用户是否在底部附近（阈值为30px）
  const isNearBottom = () => {
    if (!contentRef.current) return true;
    const { scrollTop, scrollHeight, clientHeight } = contentRef.current;
    return scrollHeight - scrollTop - clientHeight < 30;
  };

  // 滚动到底部的函数
  const scrollToBottom = () => {
    if (contentRef.current && !userIsScrollingRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  };

  // 用户滚动检测
  useEffect(() => {
    if (!collapsed && contentRef.current) {
      const handleScroll = () => {
        if (!contentRef.current) return;
        
        // 检测用户是否主动滚动离开底部
        if (!isNearBottom()) {
          userIsScrollingRef.current = true;
          // 用户滚动后3秒内不自动滚动
          clearTimeout(autoScrollTimeoutRef.current);
          autoScrollTimeoutRef.current = setTimeout(() => {
            // 只有在用户回到底部附近时才重新启用自动滚动
            if (isNearBottom()) {
              userIsScrollingRef.current = false;
            }
          }, 3000);
        } else {
          // 用户在底部附近，允许自动滚动
          userIsScrollingRef.current = false;
        }
      };

      contentRef.current.addEventListener('scroll', handleScroll, { passive: true });
      
      return () => {
        contentRef.current?.removeEventListener('scroll', handleScroll);
        clearTimeout(autoScrollTimeoutRef.current);
      };
    }
  }, [collapsed]);

  // 内容变化时的响应
  useEffect(() => {
    if (!collapsed && contentRef.current) {
      lastContentUpdateRef.current = Date.now();
      
      // 只有在用户没有主动滚动或在底部附近时才自动滚动
      if (!userIsScrollingRef.current || isNearBottom()) {
        userIsScrollingRef.current = false; // 重置状态
        requestAnimationFrame(scrollToBottom);
      }
    }
  }, [children, collapsed]);

  // 智能自动滚动机制 - 只在内容活跃更新时启用
  useEffect(() => {
    if (!collapsed && contentRef.current) {
      // 1. MutationObserver 响应内容变化
      const observer = new MutationObserver(() => {
        lastContentUpdateRef.current = Date.now();
        // 只有在允许的情况下才滚动
        if (!userIsScrollingRef.current || isNearBottom()) {
          requestAnimationFrame(scrollToBottom);
        }
      });

      observer.observe(contentRef.current, {
        childList: true,
        subtree: true,
        characterData: true
      });

      // 2. 定时检查 - 降低频率，只在内容活跃时滚动
      const checkInterval = setInterval(() => {
        const timeSinceLastUpdate = Date.now() - lastContentUpdateRef.current;
        // 只有在最近500ms内有内容更新时才继续自动滚动
        if (timeSinceLastUpdate < 500 && (!userIsScrollingRef.current || isNearBottom())) {
          scrollToBottom();
        }
      }, 100); // 降低到100ms间隔

      // 3. ResizeObserver 监听容器大小变化
      const resizeObserver = new ResizeObserver(() => {
        if (!userIsScrollingRef.current || isNearBottom()) {
          scrollToBottom();
        }
      });
      resizeObserver.observe(contentRef.current);

      return () => {
        observer.disconnect();
        resizeObserver.disconnect();
        clearInterval(checkInterval);
      };
    }
  }, [collapsed]);

  return (
    <div className="think-block my-2">
      <div
        className="think-toggle cursor-pointer text-xs text-gray-500 select-none mb-1 flex items-center"
        onClick={() => setCollapsed((c) => !c)}
      >
        <span className="mr-1">💡</span>
        <span>{collapsed ? 'Show tool reasoning / intermediate results' : 'Hide tool reasoning / intermediate results'}</span>
        <svg className={`ml-1 w-3 h-3 transition-transform ${collapsed ? '' : 'rotate-90'}`} viewBox="0 0 8 8"><path d="M2 2l2 2 2-2" stroke="currentColor" strokeWidth="1.2" fill="none"/></svg>
      </div>
      <div 
        ref={contentRef}
        className={`think-content bg-gray-50 border border-gray-200 rounded p-2 text-xs font-mono whitespace-pre-wrap transition-all duration-200 ${collapsed ? 'max-h-0 overflow-hidden' : 'max-h-96 overflow-y-auto'}`}
        style={{marginTop: collapsed ? 0 : 4}}
      >
        {children}
      </div>
    </div>
  );
}

// 辅助：将<think>标签内容替换为ThinkBlock组件
function renderWithThinkBlocks(content: string) {
  const parts: React.ReactNode[] = [];
  let lastIdx = 0;
  const regex = /<think>([\s\S]*?)<\/think>/g;
  let match;
  let idx = 0;
  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIdx) {
      parts.push(content.slice(lastIdx, match.index));
    }
    parts.push(<ThinkBlock key={idx++}>{match[1]}</ThinkBlock>);
    lastIdx = regex.lastIndex;
  }
  // 检查是否有未闭合的think块
  const openIdx = content.lastIndexOf('<think>');
  const closeIdx = content.lastIndexOf('</think>');
  if (openIdx > closeIdx) {
    // 有未闭合的think块
    parts.push(
      <ThinkBlock key={idx++}>
        {content.slice(openIdx + 7)}
      </ThinkBlock>
    );
  } else if (lastIdx < content.length) {
    parts.push(content.slice(lastIdx));
  }
  return parts;
}

export function ChatInterface({ projectCid, messages, onMessagesChange, onClearMessages }: ChatInterfaceProps) {
  const { data: project } = useProject(projectCid);
  const {
    isStreaming,
    isLoading,
    sendStreamingMessage,
  } = useChat(projectCid, messages, onMessagesChange);
  
  const [input, setInput] = useState('');
  const [dataLimit, setDataLimit] = useState(10);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);


  const scrollToBottom = (immediate: boolean = false) => {
    const behavior = immediate ? 'auto' : 'smooth';
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior });
    }, immediate ? 0 : 50);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // More aggressive scrolling during streaming to keep up with content updates
  useEffect(() => {
    if (isStreaming) {
      // Immediate scroll when streaming starts
      scrollToBottom(true);
      
      // Then continuous smooth scrolling
      const interval = setInterval(() => scrollToBottom(true), 100);
      return () => clearInterval(interval);
    }
  }, [isStreaming]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || isStreaming) return;

    const message = input.trim();
    setInput('');
    
    await sendStreamingMessage(message, dataLimit);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Use project-specific suggested questions or fallback to defaults
  const suggestedQuestions = project?.suggested_questions || [
    "What types of data can I query from this project?",
    "Show me a sample GraphQL query",
    "What entities are available in this schema?",
    "How can I filter the data?",
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold flex items-center">
              <MessageCircle className="w-5 h-5 mr-2" />
              Chat with {project?.domain_name || formatCid(projectCid)}
            </h3>
            <p className="text-sm text-gray-500">
              Ask questions about the indexed data in this project
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={onClearMessages}
              className="btn-ghost btn-sm"
              title="Clear chat history"
              disabled={messages.length === 0}
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-8">
            <MessageCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h4 className="text-lg font-semibold mb-2">Start a conversation</h4>
            <p className="text-sm text-gray-500 mb-6">
              Ask questions about the data indexed by this SubQuery project
            </p>
            
            <div className="space-y-2 max-w-md mx-auto">
              <p className="text-sm font-medium text-left">Try asking:</p>
              {suggestedQuestions.map((question: string, index: number) => (
                <button
                  key={index}
                  onClick={() => setInput(question)}
                  className="block w-full text-left p-3 text-sm bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
                >
                  "{question}"
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100'
                }`}
              >
                <div className="whitespace-pre-wrap break-words">
                  {message.role === 'assistant'
                    ? renderWithThinkBlocks(message.content)
                    : message.content}
                </div>
                <div
                  className={`text-xs mt-2 opacity-70 ${
                    message.role === 'user' ? 'text-white' : 'text-gray-500'
                  }`}
                >
                  {formatTimestamp(message.timestamp)}
                </div>
              </div>
            </div>
          ))
        )}
        
        {/* Streaming indicator */}
        {isStreaming && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-2 flex items-center">
              <Loader2 className="w-4 h-4 animate-spin mr-2" />
              <span className="text-sm">Thinking...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 p-4">
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="flex space-x-2">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about this project's data..."
                className="textarea min-h-[60px] max-h-[200px] resize-none pr-12"
                disabled={isLoading || isStreaming}
                rows={2}
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading || isStreaming}
                className="absolute bottom-2 right-2 btn-primary btn-sm p-2"
              >
                {isLoading || isStreaming ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
          
          {/* Data Limit Control */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2">
              <label className="text-gray-600 font-medium text-xs">
                Limit:
              </label>
              <div className="flex items-center space-x-2">
                <span className="text-gray-400 text-xs">1</span>
                <input
                  type="range"
                  min="1"
                  max="50"
                  value={dataLimit}
                  onChange={(e) => setDataLimit(parseInt(e.target.value))}
                  className="w-20 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <span className="text-gray-400 text-xs">50</span>
                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium min-w-fit">
                  {dataLimit}
                </span>
              </div>
            </div>
            <div className="text-gray-500 text-xs">
              Press Enter to send, Shift+Enter for new line
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}