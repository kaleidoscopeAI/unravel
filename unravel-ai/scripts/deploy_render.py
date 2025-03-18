#!/usr/bin/env python3
"""
Unravel AI - Render Deployment Script
Automates the deployment of Unravel AI to Render.com
"""

import os
import sys
import json
import argparse
import subprocess
import logging
import requests
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class RenderDeployer:
    """Handles deployment to Render.com"""
    
    def __init__(self, api_key: str):
        """
        Initialize the deployer
        
        Args:
            api_key: Render API key
        """
        self.api_key = api_key
        self.api_base_url = "https://api.render.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_web_service(self, name: str, repo_url: str, branch: str = "main",
                          env_vars: Dict[str, str] = None, plan: str = "starter") -> Dict[str, Any]:
        """
        Create a new web service on Render
        
        Args:
            name: Service name
            repo_url: GitHub repository URL
            branch: Repository branch
            env_vars: Environment variables
            plan: Render plan (starter, standard, etc.)
            
        Returns:
            Response from Render API
        """
        endpoint = f"{self.api_base_url}/services"
        
        payload = {
            "type": "web_service",
            "name": name,
            "env": "python",
            "plan": plan,
            "region": "ohio",
            "branch": branch,
            "repo": repo_url,
            "autoDeploy": "yes",
            "envVars": [{"key": k, "value": v} for k, v in (env_vars or {}).items()]
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        
        if response.status_code != 201:
            logger.error(f"Failed to create web service: {response.text}")
            return None
        
        return response.json()
    
    def create_database(self, name: str, plan: str = "starter") -> Dict[str, Any]:
        """
        Create a new PostgreSQL database on Render
        
        Args:
            name: Database name
            plan: Render plan
            
        Returns:
            Response from Render API
        """
        endpoint = f"{self.api_base_url}/databases"
        
        payload = {
            "name": name,
            "engine": "postgres",
            "version": "16",
            "region": "ohio",
            "plan": plan,
            "ipAllowList": [
                {
                    "source": "0.0.0.0/0",
                    "description": "everywhere"
                }
            ]
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        
        if response.status_code != 201:
            logger.error(f"Failed to create database: {response.text}")
            return None
        
        return response.json()
    
    def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """
        Get service deployment status
        
        Args:
            service_id: Render service ID
            
        Returns:
            Service status
        """
        endpoint = f"{self.api_base_url}/services/{service_id}"
        
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to get service status: {response.text}")
            return None
        
        return response.json()
    
    def trigger_deploy(self, service_id: str) -> bool:
        """
        Trigger a manual deployment
        
        Args:
            service_id: Render service ID
            
        Returns:
            Success status
        """
        endpoint = f"{self.api_base_url}/services/{service_id}/deploys"
        
        response = requests.post(endpoint, headers=self.headers)
        
        if response.status_code != 201:
            logger.error(f"Failed to trigger deployment: {response.text}")
            return False
        
        logger.info(f"Deployment triggered for service {service_id}")
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy Unravel AI to Render.com")
    parser.add_argument("--api-key", required=True, help="Render API key")
    parser.add_argument("--repo-url", required=True, help="GitHub repository URL")
    parser.add_argument("--branch", default="main", help="Repository branch")
    parser.add_argument("--env-file", help="Path to environment variables file (.env)")
    parser.add_argument("--service-name", default="unravel-ai", help="Service name")
    parser.add_argument("--db-name", default="unravel-ai-db", help="Database name")
    parser.add_argument("--plan", default="starter", help="Render plan")
    
    args = parser.parse_args()
    
    # Load environment variables
    env_vars = {}
    if args.env_file:
        try:
            with open(args.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        except Exception as e:
            logger.error(f"Failed to load environment variables: {str(e)}")
            sys.exit(1)
    
    # Initialize deployer
    deployer = RenderDeployer(args.api_key)
    
    # Create database
    logger.info(f"Creating database {args.db_name}...")
    db_result = deployer.create_database(args.db_name, args.plan)
    
    if not db_result:
        logger.error("Failed to create database")
        sys.exit(1)
    
    db_id = db_result["id"]
    logger.info(f"Database created with ID: {db_id}")
    
    # Wait for database to be created (in a real script, add actual waiting logic)
    logger.info("Waiting for database to be provisioned... (this may take a few minutes)")
    
    # Add database URL to environment variables
    # Note: In a real implementation, you would get this from the Render API
    # For now, we'll use a placeholder
    env_vars["DATABASE_URL"] = f"${{RENDER_DATABASE_URL}}"
    
    # Create web service
    logger.info(f"Creating web service {args.service_name}...")
    service_result = deployer.create_web_service(
        args.service_name,
        args.repo_url,
        args.branch,
        env_vars,
        args.plan
    )
    
    if not service_result:
        logger.error("Failed to create web service")
        sys.exit(1)
    
    service_id = service_result["id"]
    logger.info(f"Web service created with ID: {service_id}")
    
    logger.info("Deployment initiated. Monitor the status on your Render dashboard.")

if __name__ == "__main__":
    main()
