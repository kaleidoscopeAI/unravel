#!/usr/bin/env python3
"""
Unravel AI - Session Migration Tool
Exports and imports session state between conversations for continuous development
"""

import os
import sys
import json
import argparse
import logging
import base64
import zlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SessionManager:
    """Manages session state for Unravel AI development"""
    
    def __init__(self, session_dir: str = ".sessions"):
        """
        Initialize the session manager
        
        Args:
            session_dir: Directory to store session data
        """
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
    
    def export_session(self, name: str, data: Dict[str, Any]) -> str:
        """
        Export session data
        
        Args:
            name: Session name
            data: Session data
            
        Returns:
            Session file path
        """
        session_path = os.path.join(self.session_dir, f"{name}.json")
        
        # Save the session data
        with open(session_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Create compressed version for easier sharing
        compressed_data = zlib.compress(json.dumps(data).encode('utf-8'))
        compressed_b64 = base64.b64encode(compressed_data).decode('utf-8')
        
        compressed_path = os.path.join(self.session_dir, f"{name}.b64")
        with open(compressed_path, 'w') as f:
            f.write(compressed_b64)
        
        logger.info(f"Exported session to {session_path}")
        logger.info(f"Compressed session to {compressed_path}")
        
        return session_path
    
    def import_session(self, path_or_data: str) -> Dict[str, Any]:
        """
        Import session data
        
        Args:
            path_or_data: Path to session file or compressed data
            
        Returns:
            Session data
        """
        # Check if input is a file path
        if os.path.exists(path_or_data):
            logger.info(f"Importing session from file: {path_or_data}")
            
            # Determine file type
            if path_or_data.endswith('.json'):
                with open(path_or_data, 'r') as f:
                    data = json.load(f)
            elif path_or_data.endswith('.b64'):
                with open(path_or_data, 'r') as f:
                    compressed_b64 = f.read()
                compressed_data = base64.b64decode(compressed_b64)
                data_str = zlib.decompress(compressed_data).decode('utf-8')
                data = json.loads(data_str)
            else:
                logger.error("Unsupported file format")
                raise ValueError("Unsupported file format")
        else:
            # Try to interpret as compressed data
            try:
                compressed_data = base64.b64decode(path_or_data)
                data_str = zlib.decompress(compressed_data).decode('utf-8')
                data = json.loads(data_str)
                logger.info("Imported session from compressed data")
            except Exception as e:
                logger.error(f"Failed to import session: {str(e)}")
                raise ValueError("Invalid session data")
        
        return data
    
    def list_sessions(self) -> List[str]:
        """
        List available sessions
        
        Returns:
            List of session names
        """
        sessions = []
        
        for file in os.listdir(self.session_dir):
            if file.endswith('.json'):
                sessions.append(file[:-5])  # Remove .json extension
        
        return sessions

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unravel AI Session Migration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export session")
    export_parser.add_argument("name", help="Session name")
    export_parser.add_argument("--file", help="JSON file to export (default: stdin)")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import session")
    import_parser.add_argument("path", help="Path to session file or compressed data")
    
    # List command
    subparsers.add_parser("list", help="List available sessions")
    
    args = parser.parse_args()
    
    manager = SessionManager()
    
    if args.command == "export":
        # Read data from file or stdin
        if args.file:
            with open(args.file, 'r') as f:
                data = json.load(f)
        else:
            data = json.loads(sys.stdin.read())
        
        # Export session
        session_path = manager.export_session(args.name, data)
        print(f"Session exported to {session_path}")
        
    elif args.command == "import":
        # Import session
        data = manager.import_session(args.path)
        
        # Output data to stdout
        print(json.dumps(data, indent=2))
        
    elif args.command == "list":
        # List sessions
        sessions = manager.list_sessions()
        
        if sessions:
            print("Available sessions:")
            for session in sessions:
                print(f"  {session}")
        else:
            print("No sessions available")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
