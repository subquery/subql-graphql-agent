import React, { useState, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ProjectRegistration } from './components/ProjectRegistration';
import { ProjectList } from './components/ProjectList';
import { ProjectConfig } from './components/ProjectConfig';
import { ChatInterface } from './components/ChatInterface';
import { HealthStatus } from './components/HealthStatus';
import { Database, MessageCircle, Settings } from 'lucide-react';
import type { ChatMessage } from './types';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 30000, // 30 seconds
    },
  },
});

function AppContent() {
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'chat' | 'config'>('chat');
  // Store chat messages per project
  const [projectChatMessages, setProjectChatMessages] = useState<Record<string, ChatMessage[]>>({});

  const updateProjectMessages = useCallback((projectCid: string, messagesOrUpdater: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => {
    setProjectChatMessages(prev => {
      const currentMessages = prev[projectCid] || [];
      const newMessages = typeof messagesOrUpdater === 'function' 
        ? messagesOrUpdater(currentMessages)
        : messagesOrUpdater;
      
      return {
        ...prev,
        [projectCid]: newMessages
      };
    });
  }, []);

  const clearProjectMessages = useCallback((projectCid: string) => {
    setProjectChatMessages(prev => ({
      ...prev,
      [projectCid]: []
    }));
  }, []);

  const tabs = [
    { id: 'chat' as const, label: 'Chat', icon: MessageCircle },
    { id: 'config' as const, label: 'Configuration', icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      {/* Header */}
      <header className="border-b bg-white/95 backdrop-blur">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Database className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold">SubQuery GraphQL Agent</h1>
                <p className="text-sm text-gray-500">
                  Multi-project GraphQL query interface
                </p>
              </div>
            </div>
            
            {/* Server Status */}
            <HealthStatus />
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-120px)]">
          {/* Sidebar - Project Management */}
          <div className="lg:col-span-1 space-y-6 overflow-y-auto">
            <ProjectRegistration 
              onSuccess={() => {
                // Optionally select the newly registered project
              }}
            />
            
            <ProjectList
              selectedProject={selectedProject}
              onSelectProject={(cid) => {
                setSelectedProject(cid);
                // Switch to chat tab when selecting a project
                setActiveTab('chat');
              }}
            />
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 flex flex-col">
            {selectedProject ? (
              <>
                {/* Tabs */}
                <div className="border-b mb-6">
                  <nav className="flex space-x-8">
                    {tabs.map((tab) => {
                      const Icon = tab.icon;
                      return (
                        <button
                          key={tab.id}
                          onClick={() => setActiveTab(tab.id)}
                          className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                            activeTab === tab.id
                              ? 'border-blue-600 text-blue-600'
                              : 'border-transparent text-gray-500 hover:text-gray-900 hover:border-gray-300'
                          }`}
                        >
                          <Icon className="w-4 h-4" />
                          <span>{tab.label}</span>
                        </button>
                      );
                    })}
                  </nav>
                </div>

                {/* Tab Content */}
                <div className="flex-1 overflow-hidden">
                  {activeTab === 'chat' ? (
                    <div className="h-full border rounded-lg overflow-hidden">
                      <ChatInterface 
                        projectCid={selectedProject}
                        messages={projectChatMessages[selectedProject] || []}
                        onMessagesChange={(messagesOrUpdater) => updateProjectMessages(selectedProject, messagesOrUpdater)}
                        onClearMessages={() => clearProjectMessages(selectedProject)}
                      />
                    </div>
                  ) : (
                    <div className="overflow-y-auto h-full">
                      <ProjectConfig 
                        cid={selectedProject} 
                        onProjectDeleted={() => setSelectedProject('')}
                      />
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center max-w-md">
                  <Database className="w-16 h-16 text-gray-400 mx-auto mb-6" />
                  <h2 className="text-2xl font-semibold mb-4">Select a Project</h2>
                  <p className="text-gray-500 mb-6">
                    Choose a project from the sidebar to start chatting with its GraphQL agent
                    or register a new project to get started.
                  </p>
                  <div className="text-sm text-gray-500">
                    <p className="mb-2">💡 <strong>New to SubQuery?</strong></p>
                    <p>
                      Register your project by providing the IPFS CID of your SubQuery 
                      project manifest. The system will automatically fetch your schema 
                      and set up a specialized GraphQL agent.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App;