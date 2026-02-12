import React from 'react';
import { LayoutDashboard, Settings, History, ShieldCheck, Database, ChevronLeft, ChevronRight } from 'lucide-react';
import { ViewMode } from '../types';

interface SidebarProps {
  currentView: ViewMode;
  onViewChange: (view: ViewMode) => void;
  isCollapsed: boolean;
  onToggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ currentView, onViewChange, isCollapsed, onToggle }) => {
  
  const navItemClass = (view: ViewMode) => 
    `flex items-center ${isCollapsed ? 'justify-center px-2' : 'px-3'} py-2.5 rounded-lg font-medium transition-colors cursor-pointer group relative ${
      currentView === view 
        ? 'bg-blue-50 text-google-blue' 
        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
    }`;

  // Tooltip component for collapsed state
  const Tooltip = ({ text }: { text: string }) => (
    <div className={`absolute left-full ml-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none ${!isCollapsed ? 'hidden' : ''}`}>
      {text}
    </div>
  );

  return (
    <aside className={`${isCollapsed ? 'w-20' : 'w-64'} bg-white border-r border-gray-200 hidden md:flex flex-col h-screen fixed left-0 top-0 z-50 transition-all duration-300 ease-in-out`}>
      <div className={`h-16 flex items-center ${isCollapsed ? 'justify-center' : 'px-6'} border-b border-gray-100 relative`}>
        <ShieldCheck className="text-google-blue flex-shrink-0" size={28} />
        <div className={`ml-2 overflow-hidden transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>
           <span className="font-bold text-xl text-gray-800 tracking-tight whitespace-nowrap">BearingGuard</span>
        </div>
        
        <button 
          onClick={onToggle}
          className="absolute -right-3 top-5 bg-white border border-gray-200 rounded-full p-1 text-gray-500 hover:text-google-blue shadow-sm z-50 hover:scale-110 transition-transform"
        >
          {isCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto overflow-x-hidden">
        <div 
          onClick={() => onViewChange(ViewMode.DIAGNOSTICS)} 
          className={navItemClass(ViewMode.DIAGNOSTICS)}
        >
          <LayoutDashboard size={20} className={`flex-shrink-0 ${isCollapsed ? "" : "mr-3"}`} />
          {!isCollapsed && <span className="whitespace-nowrap transition-opacity duration-200">Field Diagnostics</span>}
          <Tooltip text="Field Diagnostics" />
        </div>
        
        <div 
          onClick={() => onViewChange(ViewMode.HISTORY)} 
          className={navItemClass(ViewMode.HISTORY)}
        >
          <History size={20} className={`flex-shrink-0 ${isCollapsed ? "" : "mr-3"}`} />
          {!isCollapsed && <span className="whitespace-nowrap transition-opacity duration-200">History Log</span>}
          <Tooltip text="History Log" />
        </div>

        <div className={`px-3 pt-6 pb-2 transition-opacity duration-200 ${isCollapsed ? 'opacity-0 h-0 p-0 overflow-hidden' : 'opacity-100'}`}>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">Admin Zone</p>
        </div>
        {isCollapsed && <div className="h-4"></div>}

        <div 
          onClick={() => onViewChange(ViewMode.TRAINING)} 
          className={navItemClass(ViewMode.TRAINING)}
        >
          <Database size={20} className={`flex-shrink-0 ${isCollapsed ? "" : "mr-3"}`} />
          {!isCollapsed && <span className="whitespace-nowrap transition-opacity duration-200">Data & Training</span>}
          <Tooltip text="Data & Training" />
        </div>
      </nav>

      <div className="p-4 border-t border-gray-100">
        <div 
          onClick={() => onViewChange(ViewMode.SETTINGS)} 
          className={navItemClass(ViewMode.SETTINGS)}
        >
          <Settings size={20} className={`flex-shrink-0 ${isCollapsed ? "" : "mr-3"}`} />
          {!isCollapsed && <span className="whitespace-nowrap transition-opacity duration-200">Settings</span>}
          <Tooltip text="Settings" />
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;