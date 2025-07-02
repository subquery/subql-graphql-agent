import React, { useState, useEffect } from 'react';
import { useProject, useUpdateProjectConfig, useDeleteProject } from '../hooks/useProjects';
import { Settings, Save, X, Plus, Trash2, Loader2, AlertCircle } from 'lucide-react';
import { formatCid } from '../lib/utils';
import { ConfirmDialog } from './ConfirmDialog';

interface ProjectConfigProps {
  cid: string;
  onProjectDeleted?: () => void;
}

export function ProjectConfig({ cid, onProjectDeleted }: ProjectConfigProps) {
  const { data: project, isLoading } = useProject(cid);
  const updateConfig = useUpdateProjectConfig();
  const deleteProject = useDeleteProject();
  
  const [isEditing, setIsEditing] = useState(false);
  const [domainName, setDomainName] = useState('');
  const [capabilities, setCapabilities] = useState<string[]>([]);
  const [declineMessage, setDeclineMessage] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  useEffect(() => {
    if (project) {
      setDomainName(project.domain_name);
      setCapabilities([...project.domain_capabilities]);
      setDeclineMessage(project.decline_message);
      setEndpoint(project.endpoint);
      setSuggestedQuestions([...project.suggested_questions]);
    }
  }, [project]);

  const handleSave = async () => {
    if (!project) return;

    try {
      await updateConfig.mutateAsync({
        cid: project.cid,
        updates: {
          domain_name: domainName,
          domain_capabilities: capabilities.filter(cap => cap.trim()),
          decline_message: declineMessage,
          endpoint: endpoint,
          suggested_questions: suggestedQuestions.filter(q => q.trim()),
        },
      });
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to update project config:', error);
    }
  };

  const handleCancel = () => {
    if (project) {
      setDomainName(project.domain_name);
      setCapabilities([...project.domain_capabilities]);
      setDeclineMessage(project.decline_message);
      setEndpoint(project.endpoint);
      setSuggestedQuestions([...project.suggested_questions]);
    }
    setIsEditing(false);
  };

  const addCapability = () => {
    setCapabilities([...capabilities, '']);
  };

  const updateCapability = (index: number, value: string) => {
    const updated = [...capabilities];
    updated[index] = value;
    setCapabilities(updated);
  };

  const removeCapability = (index: number) => {
    setCapabilities(capabilities.filter((_, i) => i !== index));
  };

  const addSuggestedQuestion = () => {
    setSuggestedQuestions([...suggestedQuestions, '']);
  };

  const updateSuggestedQuestion = (index: number, value: string) => {
    const updated = [...suggestedQuestions];
    updated[index] = value;
    setSuggestedQuestions(updated);
  };

  const removeSuggestedQuestion = (index: number) => {
    setSuggestedQuestions(suggestedQuestions.filter((_, i) => i !== index));
  };

  const handleDeleteProject = async () => {
    try {
      await deleteProject.mutateAsync(cid);
      setShowDeleteDialog(false);
      onProjectDeleted?.();
    } catch (error) {
      console.error('Failed to delete project:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin mr-2" />
        <span>Loading project configuration...</span>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="text-center p-8">
        <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-red-700 mb-2">Project not found</h3>
        <p className="text-sm text-red-600">
          Could not load configuration for CID: {formatCid(cid)}
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="card-title flex items-center">
              <Settings className="w-5 h-5 mr-2" />
              Project Configuration
            </h3>
            <p className="card-description">
              CID: {formatCid(project.cid, 16)} • {project.cached ? 'Cached' : 'Not cached'}
            </p>
          </div>
          <div className="flex space-x-2">
            {!isEditing ? (
              <>
                <button
                  onClick={() => setIsEditing(true)}
                  className="btn-outline btn-sm"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Edit
                </button>
                <button
                  onClick={() => setShowDeleteDialog(true)}
                  className="btn-outline btn-sm text-red-600 border-red-200 hover:bg-red-50"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={handleSave}
                  disabled={updateConfig.isPending}
                  className="btn-primary btn-sm"
                >
                  {updateConfig.isPending ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="w-4 h-4 mr-2" />
                  )}
                  Save
                </button>
                <button
                  onClick={handleCancel}
                  disabled={updateConfig.isPending}
                  className="btn-outline btn-sm"
                >
                  <X className="w-4 h-4 mr-2" />
                  Cancel
                </button>
              </>
            )}
          </div>
        </div>
      </div>
      
      <div className="card-content space-y-6">
        {/* Domain Name */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Domain Name</label>
          {isEditing ? (
            <input
              type="text"
              value={domainName}
              onChange={(e) => setDomainName(e.target.value)}
              className="input"
              placeholder="My SubQuery Project"
            />
          ) : (
            <p className="p-3 bg-muted rounded-md">{project.domain_name}</p>
          )}
        </div>

        {/* Endpoint */}
        <div className="space-y-2">
          <label className="text-sm font-medium">GraphQL Endpoint</label>
          {isEditing ? (
            <input
              type="url"
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              className="input"
              placeholder="https://api.subquery.network/sq/..."
            />
          ) : (
            <p className="p-3 bg-muted rounded-md font-mono text-sm break-all">
              {project.endpoint}
            </p>
          )}
        </div>

        {/* Domain Capabilities */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Domain Capabilities</label>
          {isEditing ? (
            <div className="space-y-3">
              {capabilities.map((capability, index) => (
                <div key={index} className="flex space-x-2">
                  <input
                    type="text"
                    value={capability}
                    onChange={(e) => updateCapability(index, e.target.value)}
                    className="input flex-1"
                    placeholder="Capability description..."
                  />
                  <button
                    onClick={() => removeCapability(index)}
                    className="btn-outline btn-sm px-3"
                    title="Remove capability"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
              <button
                onClick={addCapability}
                className="btn-outline btn-sm"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Capability
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {project.domain_capabilities.map((capability, index) => (
                <div key={index} className="p-3 bg-muted rounded-md text-sm">
                  • {capability}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Decline Message */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Decline Message</label>
          <p className="text-xs text-muted-foreground">
            Message shown when users ask questions outside the project domain
          </p>
          {isEditing ? (
            <textarea
              value={declineMessage}
              onChange={(e) => setDeclineMessage(e.target.value)}
              className="textarea min-h-[100px]"
              placeholder="I'm specialized in this project's data queries..."
            />
          ) : (
            <p className="p-3 bg-muted rounded-md text-sm">
              {project.decline_message}
            </p>
          )}
        </div>

        {/* Suggested Questions */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Suggested Questions</label>
          <p className="text-xs text-muted-foreground">
            Sample questions that users can ask to explore the data
          </p>
          {isEditing ? (
            <div className="space-y-3">
              {suggestedQuestions.map((question, index) => (
                <div key={index} className="flex space-x-2">
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => updateSuggestedQuestion(index, e.target.value)}
                    className="input flex-1"
                    placeholder="Sample question..."
                  />
                  <button
                    onClick={() => removeSuggestedQuestion(index)}
                    className="btn-outline btn-sm px-3"
                    title="Remove question"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
              <button
                onClick={addSuggestedQuestion}
                className="btn-outline btn-sm"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Question
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {project?.suggested_questions.map((question, index) => (
                <div key={index} className="p-3 bg-muted rounded-md text-sm">
                  • {question}
                </div>
              ))}
            </div>
          )}
        </div>

        {updateConfig.error && (
          <div className="p-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center">
              <AlertCircle className="w-4 h-4 mr-2" />
              <span>
                Failed to update configuration: {' '}
                {updateConfig.error instanceof Error 
                  ? updateConfig.error.message 
                  : 'Unknown error'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        onConfirm={handleDeleteProject}
        title="Delete Project"
        message={`Are you sure you want to delete "${project?.domain_name || 'this project'}"? This action cannot be undone and will remove all project configuration.`}
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        isLoading={deleteProject.isPending}
      />
    </div>
  );
}