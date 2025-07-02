import React, { useState, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ProjectRegistration } from './components/ProjectRegistration';
import { ProjectList } from './components/ProjectList';
import { ProjectConfig } from './components/ProjectConfig';
import { ChatInterface } from './components/ChatInterface';
import { HealthStatus } from './components/HealthStatus';
import { Database, MessageCircle, Settings, HelpCircle, X, Info, Zap } from 'lucide-react';
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
  const [showBestPractices, setShowBestPractices] = useState(false);
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
            
            <div className="flex items-center space-x-3">
              {/* Best Practices Button */}
              <button
                onClick={() => setShowBestPractices(true)}
                className="flex items-center space-x-2 px-3 py-2 text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded-md transition-colors"
                title="View best practices for optimal performance"
              >
                <HelpCircle className="w-4 h-4" />
                <span>Best Practices</span>
              </button>
              
              {/* Server Status */}
              <HealthStatus />
            </div>
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

      {/* Best Practices Modal */}
      {showBestPractices && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-2">
                <HelpCircle className="w-5 h-5 text-blue-600" />
                <h2 className="text-xl font-semibold">Best Practices for Optimal Performance</h2>
              </div>
              <button
                onClick={() => setShowBestPractices(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-6 space-y-6">
              <div className="flex items-start space-x-3">
                <Database className="w-6 h-6 text-blue-600 mt-1" />
                <div>
                  <h3 className="font-semibold text-lg mb-2">Schema Documentation</h3>
                  <p className="text-gray-700 mb-3">
                    Add descriptive comments to your entity schema to explain each entity and field purpose. 
                    Well-documented schemas help the AI understand your data structure better and generate more accurate queries.
                  </p>
                  <div className="bg-gray-50 p-3 rounded-md text-sm font-mono">
                    <div className="text-gray-600">""" Example: """</div>
                    <div className="text-green-600">"""</div>
                    <div className="text-green-600">User account entity with staking information</div>
                    <div className="text-green-600">"""</div>
                    <div>type Account @entity {"{"}</div>
                    <div className="ml-4">
                      <div className="text-green-600">"""Unique wallet address"""</div>
                      <div>id: ID!</div>
                      <div className="text-green-600">"""Total staked amount in tokens"""</div>
                      <div>stakedAmount: BigInt!</div>
                    </div>
                    <div>{"}"}</div>
                  </div>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <Zap className="w-6 h-6 text-blue-600 mt-1" />
                <div>
                  <h3 className="font-semibold text-lg mb-2">Model Selection & Performance</h3>
                  <div className="space-y-2 text-gray-700">
                    <p><strong>For best accuracy:</strong> Use stronger models like GPT-4o or DeepSeek V3</p>
                    <p><strong>For faster responses:</strong> Use faster models and limit data output quantities</p>
                    <p><strong>Balance is key:</strong> Choose the right trade-off between model capability and response speed</p>
                  </div>
                  <div className="mt-3 bg-blue-50 p-3 rounded-md">
                    <div className="text-sm font-medium text-blue-900 mb-2">Model Recommendations:</div>
                    <div className="text-sm text-blue-800 space-y-1">
                      <div>• <strong>Production:</strong> GPT-4o (best reliability)</div>
                      <div>• <strong>Cost-effective:</strong> DeepSeek V3 (excellent value)</div>
                      <div>• <strong>Development:</strong> GPT-4.1-mini (budget-friendly)</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <Info className="w-6 h-6 text-blue-600 mt-1" />
                <div>
                  <h3 className="font-semibold text-lg mb-2">Query Optimization</h3>
                  <div className="space-y-2 text-gray-700">
                    <p>Use the data limit slider in chat to control result sizes. Smaller limits provide faster responses, while larger limits give more comprehensive data.</p>
                    <p>Start with smaller limits (5-10 records) for exploration, then increase as needed for detailed analysis.</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="border-t p-4 text-center">
              <button
                onClick={() => setShowBestPractices(false)}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Got it!
              </button>
            </div>
          </div>
        </div>
      )}
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