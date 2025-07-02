import React, { useState, useEffect } from 'react';
import { useProjects, useDeleteProject } from '../hooks/useProjects';
import { formatCid } from '../lib/utils';
import { Loader2, Database, CheckCircle, AlertCircle, RefreshCw, Trash2, MoreVertical } from 'lucide-react';
import { ConfirmDialog } from './ConfirmDialog';

interface ProjectListProps {
  selectedProject?: string;
  onSelectProject: (cid: string) => void;
}

export function ProjectList({ selectedProject, onSelectProject }: ProjectListProps) {
  const { data: projectsData, isLoading, error, refetch } = useProjects();
  const deleteProject = useDeleteProject();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<{ cid: string; name: string } | null>(null);
  const [expandedProject, setExpandedProject] = useState<string | null>(null);

  const handleDeleteClick = (project: { cid: string; domain_name: string }, e: React.MouseEvent) => {
    e.stopPropagation();
    setProjectToDelete({ cid: project.cid, name: project.domain_name });
    setShowDeleteDialog(true);
    setExpandedProject(null);
  };

  const handleConfirmDelete = async () => {
    if (!projectToDelete) return;
    
    try {
      await deleteProject.mutateAsync(projectToDelete.cid);
      setShowDeleteDialog(false);
      setProjectToDelete(null);
      
      // Clear selection if the deleted project was selected
      if (selectedProject === projectToDelete.cid) {
        onSelectProject('');
      }
    } catch (error) {
      console.error('Failed to delete project:', error);
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteDialog(false);
    setProjectToDelete(null);
  };

  const toggleProjectMenu = (cid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedProject(expandedProject === cid ? null : cid);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setExpandedProject(null);
    };

    if (expandedProject) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [expandedProject]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin mr-2" />
        <span>Loading projects...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-center">
        <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-red-700 mb-2">Failed to load projects</h3>
        <p className="text-sm text-red-600 mb-4">
          {error instanceof Error ? error.message : 'Unknown error occurred'}
        </p>
        <button
          onClick={() => refetch()}
          className="btn-outline btn-sm"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Retry
        </button>
      </div>
    );
  }

  const projects = projectsData?.projects || [];

  if (projects.length === 0) {
    return (
      <div className="text-center py-8">
        <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Projects</h3>
        <p className="text-sm text-gray-500">
          Register your first SubQuery project to get started
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Projects ({projects.length})</h3>
        <button
          onClick={() => refetch()}
          className="btn-ghost btn-sm"
          title="Refresh projects"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      
      <div className="space-y-2">
        {projects.map((project) => (
          <div key={project.cid} className="relative">
            <button
              onClick={() => onSelectProject(project.cid)}
              className={`w-full text-left p-4 rounded-lg border transition-colors ${
                selectedProject === project.cid
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:bg-gray-50'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0 pr-2">
                  <h4 className="font-medium truncate">{project.domain_name}</h4>
                  <p className="text-sm text-gray-500 mt-1">
                    CID: {formatCid(project.cid, 12)}
                  </p>
                  <p className="text-xs text-gray-500 mt-1 truncate">
                    {project.endpoint}
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  {project.cached ? (
                    <div className="flex items-center text-green-600">
                      <CheckCircle className="w-4 h-4 mr-1" />
                      <span className="text-xs">Cached</span>
                    </div>
                  ) : (
                    <div className="flex items-center text-gray-400">
                      <Database className="w-4 h-4 mr-1" />
                      <span className="text-xs">Not cached</span>
                    </div>
                  )}
                  <button
                    onClick={(e) => toggleProjectMenu(project.cid, e)}
                    className="p-1 rounded hover:bg-gray-200"
                    title="More options"
                  >
                    <MoreVertical className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
            </button>
            
            {/* Dropdown menu */}
            {expandedProject === project.cid && (
              <div className="absolute right-2 top-12 z-10 bg-white border rounded-md shadow-lg py-1 min-w-[120px]">
                <button
                  onClick={(e) => handleDeleteClick(project, e)}
                  className="w-full px-3 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={showDeleteDialog}
        onClose={handleCancelDelete}
        onConfirm={handleConfirmDelete}
        title="Delete Project"
        message={
          projectToDelete
            ? `Are you sure you want to delete "${projectToDelete.name}"? This action cannot be undone and will remove all project configuration.`
            : ''
        }
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        isLoading={deleteProject.isPending}
      />
    </div>
  );
}