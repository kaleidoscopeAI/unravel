#!/usr/bin/env python3
"""
Unravel AI - Graph-based Code Analysis with LLM Enhancement
Builds and analyzes code structure graphs with LLM-powered insights
"""

import os
import sys
import re
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import the LLM analyzer using relative import
try:
    from .llm_interface import UnravelAIAnalyzer
    HAS_LLM = True
except ImportError:
    logger.warning("LLM interface not found. Running without LLM enhancement.")
    HAS_LLM = False
    UnravelAIAnalyzer = None

class CodeNode:
    """Represents a node in the code graph (function, class, module, etc.)"""
    
    def __init__(self, name: str, file_path: str, node_type: str, start_line: int = 0, end_line: int = 0):
        self.name = name
        self.file_path = file_path
        self.node_type = node_type  # 'function', 'class', 'module', etc.
        self.start_line = start_line
        self.end_line = end_line
        self.properties = {}
        self.dependencies = set()
        self.callers = set()
        self.code_fragment = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "node_type": self.node_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "properties": self.properties,
            "dependencies": list(self.dependencies),
            "callers": list(self.callers),
            "code_fragment": self.code_fragment[:500] + "..." if len(self.code_fragment) > 500 else self.code_fragment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeNode':
        """Create node from dictionary"""
        node = cls(
            name=data["name"],
            file_path=data["file_path"],
            node_type=data["node_type"],
            start_line=data["start_line"],
            end_line=data["end_line"]
        )
        node.properties = data.get("properties", {})
        node.dependencies = set(data.get("dependencies", []))
        node.callers = set(data.get("callers", []))
        node.code_fragment = data.get("code_fragment", "")
        return node

class CodeGraphBuilder:
    """Builds a graph representation of code structure"""
    
    def __init__(self, llm_analyzer: Optional[UnravelAIAnalyzer] = None):
        """
        Initialize the graph builder
        
        Args:
            llm_analyzer: LLM analyzer for enhanced analysis
        """
        self.graph = nx.DiGraph()
        self.node_map = {}  # name -> CodeNode
        self.llm_analyzer = llm_analyzer
    
    def build_graph(self, decompiled_files: List[str]) -> nx.DiGraph:
        """
        Build a graph representation of the code
        
        Args:
            decompiled_files: List of paths to decompiled files
            
        Returns:
            Directed graph of code structure
        """
        # Reset the graph
        self.graph = nx.DiGraph()
        self.node_map = {}
        
        # Process files
        for file_path in decompiled_files:
            self._process_file(file_path)
        
        # Build cross-file dependencies
        self._build_cross_file_dependencies()
        
        # Enhance with LLM if available
        if self.llm_analyzer:
            self._enhance_with_llm(decompiled_files)
        
        return self.graph
    
    def _process_file(self, file_path: str) -> None:
        """
        Process a single file and add nodes to the graph
        
        Args:
            file_path: Path to the file
        """
        try:
            # Read file content
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Determine file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Create module node
            module_name = os.path.basename(file_path)
            module_node = CodeNode(module_name, file_path, "module")
            self.node_map[module_name] = module_node
            self.graph.add_node(module_name, 
                               attr_dict=module_node.to_dict(),
                               label=module_name,
                               type="module")
            
            # Process based on file type
            if file_ext in ['.py', '.pyw']:
                self._process_python_file(file_path, content, module_name)
            elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
                self._process_javascript_file(file_path, content, module_name)
            elif file_ext in ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp']:
                self._process_c_cpp_file(file_path, content, module_name)
            elif file_ext in ['.java']:
                self._process_java_file(file_path, content, module_name)
            else:
                # Generic processor for other file types
                self._process_generic_file(file_path, content, module_name)
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    def _process_python_file(self, file_path: str, content: str, module_name: str) -> None:
        """
        Process a Python file
        
        Args:
            file_path: Path to the file
            content: File content
            module_name: Module node name
        """
        # Extract imports
        import_pattern = r'^\s*(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w.]+)(?:\s+as\s+(\w+))?)'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            if match.group(1):  # from X import Y
                module = match.group(1)
                imports = [name.strip() for name in match.group(2).split(',')]
                for imp in imports:
                    self.node_map[module_name].dependencies.add(module + '.' + imp)
            else:  # import X
                module = match.group(3)
                self.node_map[module_name].dependencies.add(module)
        
        # Extract classes
        class_pattern = r'^\s*class\s+(\w+)(?:\((.*?)\))?:'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)
            full_class_name = f"{module_name}.{class_name}"
            class_node = CodeNode(full_class_name, file_path, "class")
            self.node_map[full_class_name] = class_node
            
            # Add to graph
            self.graph.add_node(full_class_name, 
                               attr_dict=class_node.to_dict(),
                               label=class_name,
                               type="class")
            
            # Add edge from module to class
            self.graph.add_edge(module_name, full_class_name, type="contains")
            
            # Process inheritance
            if match.group(2):
                inheritances = [base.strip() for base in match.group(2).split(',')]
                for base in inheritances:
                    if base != "object":
                        self.node_map[full_class_name].dependencies.add(base)
        
        # Extract functions/methods
        func_pattern = r'^\s*def\s+(\w+)\s*\((.*?)\):'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            full_func_name = f"{module_name}.{func_name}"
            func_node = CodeNode(full_func_name, file_path, "function")
            self.node_map[full_func_name] = func_node
            
            # Add to graph
            self.graph.add_node(full_func_name, 
                               attr_dict=func_node.to_dict(),
                               label=func_name,
                               type="function")
            
            # Add edge from module to function
            self.graph.add_edge(module_name, full_func_name, type="contains")
    
    def _process_javascript_file(self, file_path: str, content: str, module_name: str) -> None:
        """Process a JavaScript/TypeScript file"""
        # Extract imports (ES6)
        import_pattern = r'^\s*import\s+(?:{(.*?)}|(\w+))\s+from\s+[\'"](.+?)[\'"]'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            if match.group(1):  # import { ... } from '...'
                imports = [name.strip() for name in match.group(1).split(',')]
                module = match.group(3)
                for imp in imports:
                    self.node_map[module_name].dependencies.add(module + '.' + imp)
            else:  # import X from '...'
                module = match.group(3)
                self.node_map[module_name].dependencies.add(module)
        
        # Extract classes
        class_pattern = r'^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)
            full_class_name = f"{module_name}.{class_name}"
            class_node = CodeNode(full_class_name, file_path, "class")
            self.node_map[full_class_name] = class_node
            
            # Add to graph
            self.graph.add_node(full_class_name, 
                               attr_dict=class_node.to_dict(),
                               label=class_name,
                               type="class")
            
            # Add edge from module to class
            self.graph.add_edge(module_name, full_class_name, type="contains")
            
            # Process inheritance
            if match.group(2):
                base = match.group(2).strip()
                self.node_map[full_class_name].dependencies.add(base)
        
        # Extract functions
        func_pattern = r'^\s*(?:export\s+)?(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:function|\(.*?\)\s*=>)|(?:\(.*?\)))'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            full_func_name = f"{module_name}.{func_name}"
            func_node = CodeNode(full_func_name, file_path, "function")
            self.node_map[full_func_name] = func_node
            
            # Add to graph
            self.graph.add_node(full_func_name, 
                               attr_dict=func_node.to_dict(),
                               label=func_name,
                               type="function")
            
            # Add edge from module to function
            self.graph.add_edge(module_name, full_func_name, type="contains")
    
    def _process_c_cpp_file(self, file_path: str, content: str, module_name: str) -> None:
        """Process a C/C++ file"""
        # Extract includes
        include_pattern = r'^\s*#include\s+[<"](.+?)[>"]'
        for match in re.finditer(include_pattern, content, re.MULTILINE):
            include = match.group(1)
            self.node_map[module_name].dependencies.add(include)
        
        # Extract classes (C++)
        class_pattern = r'^\s*(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|protected|private)\s+(\w+))?'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)
            full_class_name = f"{module_name}.{class_name}"
            class_node = CodeNode(full_class_name, file_path, "class")
            self.node_map[full_class_name] = class_node
            
            # Add to graph
            self.graph.add_node(full_class_name, 
                               attr_dict=class_node.to_dict(),
                               label=class_name,
                               type="class")
            
            # Add edge from module to class
            self.graph.add_edge(module_name, full_class_name, type="contains")
            
            # Process inheritance
            if match.group(2):
                base = match.group(2).strip()
                self.node_map[full_class_name].dependencies.add(base)
        
        # Extract functions
        func_pattern = r'^\s*(?:\w+\s+)+(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*(?:{\s*)?$'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            # Skip common C/C++ keywords that might match our pattern
            if func_name in ["if", "for", "while", "switch", "return"]:
                continue
                
            full_func_name = f"{module_name}.{func_name}"
            func_node = CodeNode(full_func_name, file_path, "function")
            self.node_map[full_func_name] = func_node
            
            # Add to graph
            self.graph.add_node(full_func_name, 
                               attr_dict=func_node.to_dict(),
                               label=func_name,
                               type="function")
            
            # Add edge from module to function
            self.graph.add_edge(module_name, full_func_name, type="contains")
    
    def _process_java_file(self, file_path: str, content: str, module_name: str) -> None:
        """Process a Java file"""
        # Extract package and imports
        package_pattern = r'^\s*package\s+([\w.]+)\s*;'
        package_match = re.search(package_pattern, content)
        package = package_match.group(1) if package_match else ""
        
        import_pattern = r'^\s*import\s+([\w.]+)(?:\s*;\s*|\.(?:\*|\w+)\s*;)'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            imp = match.group(1)
            self.node_map[module_name].dependencies.add(imp)
        
        # Extract classes
        class_pattern = r'^\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)
            full_class_name = f"{package}.{class_name}" if package else f"{module_name}.{class_name}"
            class_node = CodeNode(full_class_name, file_path, "class")
            self.node_map[full_class_name] = class_node
            
            # Add to graph
            self.graph.add_node(full_class_name, 
                               attr_dict=class_node.to_dict(),
                               label=class_name,
                               type="class")
            
            # Add edge from module to class
            self.graph.add_edge(module_name, full_class_name, type="contains")
            
            # Process inheritance
            if match.group(2):
                base = match.group(2).strip()
                self.node_map[full_class_name].dependencies.add(base)
            
            # Process interfaces
            if match.group(3):
                interfaces = [intf.strip() for intf in match.group(3).split(',')]
                for intf in interfaces:
                    self.node_map[full_class_name].dependencies.add(intf)
        
        # Extract methods
        method_pattern = r'^\s*(?:public|private|protected)?\s*(?:static|final|abstract)?\s*(?:[\w<>\[\],\s]+)\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(method_pattern, content, re.MULTILINE):
            method_name = match.group(1)
            # Skip constructors (same name as class)
            if any(method_name == os.path.basename(class_name) for class_name in self.node_map.keys() if self.node_map[class_name].node_type == "class"):
                continue
                
            full_method_name = f"{module_name}.{method_name}"
            method_node = CodeNode(full_method_name, file_path, "method")
            self.node_map[full_method_name] = method_node
            
            # Add to graph
            self.graph.add_node(full_method_name, 
                               attr_dict=method_node.to_dict(),
                               label=method_name,
                               type="method")
            
            # Add edge from module to method
            self.graph.add_edge(module_name, full_method_name, type="contains")
    
    def _process_generic_file(self, file_path: str, content: str, module_name: str) -> None:
        """Generic processor for other file types"""
        # Placeholder for future implementation
        pass
    
    def _build_cross_file_dependencies(self) -> None:
        """Build cross-file dependencies between nodes"""
        # For each node, check its dependencies and add edges
        for node_name, node in self.node_map.items():
            for dep_name in node.dependencies:
                # Check if the dependency exists in our node map
                for potential_dep in self.node_map:
                    if potential_dep.endswith(dep_name):
                        # Add edge from node to dependency
                        self.graph.add_edge(node_name, potential_dep, type="depends")
                        break
    
    def _enhance_with_llm(self, decompiled_files: List[str]) -> None:
        """
        Enhance the graph with LLM-powered insights
        
        Args:
            decompiled_files: List of paths to decompiled files
        """
        if not self.llm_analyzer:
            return
        
        logger.info("Enhancing code graph with LLM...")
        
        try:
            # Analyze the code as a whole
            analysis_results = self.llm_analyzer.analyze_software(
                decompiled_files, 
                analysis_type="custom:Analyze these files and identify key components, their relationships, and dependencies. Focus on the architectural structure."
            )
            
            # Process results
            if "_summary" in analysis_results:
                summary = analysis_results["_summary"]
                # Add summary as a property to the graph
                self.graph.graph["llm_analysis"] = summary
            
            # Analyze individual components/classes/functions
            for node_name, node in self.node_map.items():
                if node.node_type in ["class", "function", "method"]:
                    # Get the code fragment for this node
                    if node.code_fragment:
                        try:
                            result = self.llm_analyzer.analyze_code(
                                {node.name: node.code_fragment},
                                analysis_type="custom:Analyze this code component. Identify its purpose, complexity, potential issues, and relationships to other components."
                            )
                            
                            # Add the analysis as a property to the node
                            self.node_map[node_name].properties["llm_analysis"] = result
                            self.graph.nodes[node_name]["llm_analysis"] = result
                        except Exception as e:
                            logger.error(f"Error analyzing node {node_name}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error during LLM enhancement: {str(e)}")
    
    def save_graph(self, output_path: str) -> None:
        """
        Save the graph to a file
        
        Args:
            output_path: Path to save the graph
        """
        # Convert the graph to a serializable format
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node].copy()
            node_data["id"] = node
            graph_data["nodes"].append(node_data)
        
        for edge in self.graph.edges():
            edge_data = self.graph.edges[edge].copy()
            edge_data["source"] = edge[0]
            edge_data["target"] = edge[1]
            graph_data["edges"].append(edge_data)
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Graph saved to {output_path}")
    
    def visualize_graph(self, output_path: str = None) -> None:
        """
        Visualize the graph
        
        Args:
            output_path: Path to save the visualization
        """
        try:
            # Create a new figure
            plt.figure(figsize=(12, 10))
            
            # Create a copy of the graph for visualization
            vis_graph = self.graph.copy()
            
            # Define node colors based on type
            color_map = {
                "module": "blue",
                "class": "green",
                "function": "red",
                "method": "purple"
            }
            
            # Set node colors
            node_colors = [
                color_map.get(vis_graph.nodes[node].get("type", ""), "gray") 
                for node in vis_graph.nodes()
            ]
            
            # Set node sizes based on importance
            node_sizes = [
                100 + 50 * len(vis_graph.edges(node)) + 20 * len(list(vis_graph.predecessors(node)))
                for node in vis_graph.nodes()
            ]
            
            # Set edge colors based on type
            edge_colors = [
                "green" if vis_graph.edges[edge].get("type") == "contains" else
                "red" if vis_graph.edges[edge].get("type") == "depends" else
                "blue"
                for edge in vis_graph.edges()
            ]
            
            # Create positions using a spring layout
            pos = nx.spring_layout(vis_graph, k=0.15, iterations=50)
            
            # Draw the graph
            nx.draw_networkx_nodes(vis_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
            nx.draw_networkx_edges(vis_graph, pos, edge_color=edge_colors, width=1.0, alpha=0.5, arrows=True)
            nx.draw_networkx_labels(vis_graph, pos, font_size=8)
            
            plt.title("Code Structure Graph")
            plt.axis("off")
            
            # Save or show the graph
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Graph visualization saved to {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")

def analyze_codebase(decompiled_dir: str, output_dir: str, use_llm: bool = True) -> None:
    """
    Analyze a codebase and generate a code structure graph
    
    Args:
        decompiled_dir: Directory containing decompiled files
        output_dir: Directory to save the results
        use_llm: Whether to use LLM enhancement
    """
    logger.info(f"Analyzing codebase in {decompiled_dir}...")
    
    # Collect decompiled files
    decompiled_files = []
    for root, _, files in os.walk(decompiled_dir):
        for file in files:
            file_path = os.path.join(root, file)
            decompiled_files.append(file_path)
    
    logger.info(f"Found {len(decompiled_files)} files for analysis")
    
    # Initialize LLM analyzer if requested
    llm_analyzer = None
    if use_llm and HAS_LLM:
        try:
            llm_analyzer = UnravelAIAnalyzer()
            llm_analyzer.setup()
            logger.info("LLM analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing LLM analyzer: {str(e)}")
    
    # Build code graph
    graph_builder = CodeGraphBuilder(llm_analyzer)
    graph = graph_builder.build_graph(decompiled_files)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graph data
    graph_path = os.path.join(output_dir, "code_graph.json")
    graph_builder.save_graph(graph_path)
    
    # Visualize graph
    vis_path = os.path.join(output_dir, "code_graph.png")
    graph_builder.visualize_graph(vis_path)
    
    # Generate summary report
    report_path = os.path.join(output_dir, "code_analysis_report.md")
    generate_report(graph, report_path, llm_analyzer)
    
    # Clean up
    if llm_analyzer:
        llm_analyzer.cleanup()
    
    logger.info("Code analysis complete")

def generate_report(graph: nx.DiGraph, output_path: str, llm_analyzer: Optional[UnravelAIAnalyzer] = None) -> None:
    """
    Generate a report from the code graph
    
    Args:
        graph: Code graph
        output_path: Path to save the report
        llm_analyzer: LLM analyzer for enhanced reporting
    """
    with open(output_path, 'w') as f:
        f.write("# Code Analysis Report\n\n")
        
        # Graph statistics
        f.write("## Graph Statistics\n\n")
        f.write(f"- Total components: {graph.number_of_nodes()}\n")
        f.write(f"- Total relationships: {graph.number_of_edges()}\n")
        
        # Component types
        modules = [n for n, d in graph.nodes(data=True) if d.get("type") == "module"]
        classes = [n for n, d in graph.nodes(data=True) if d.get("type") == "class"]
        functions = [n for n, d in graph.nodes(data=True) if d.get("type") in ["function", "method"]]
        
        f.write(f"- Modules: {len(modules)}\n")
        f.write(f"- Classes: {len(classes)}\n")
        f.write(f"- Functions/Methods: {len(functions)}\n\n")
        
        # Key components (high centrality)
        f.write("## Key Components\n\n")
        centrality = nx.degree_centrality(graph)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        for i, (node, score) in enumerate(sorted_centrality[:10]):
            node_type = graph.nodes[node].get("type", "unknown")
            f.write(f"{i+1}. **{node}** (Type: {node_type}, Centrality: {score:.3f})\n")
            
            # Add LLM analysis if available
            if "llm_analysis" in graph.nodes[node]:
                analysis = graph.nodes[node]["llm_analysis"]
                f.write(f"   - Analysis: {analysis[:200]}...\n")
            
            f.write("\n")
        
        # Overall analysis
        f.write("## Overall Analysis\n\n")
        
        if "llm_analysis" in graph.graph:
            f.write(graph.graph["llm_analysis"])
        elif llm_analyzer:
            try:
                # Generate an overall analysis
                analysis = "No overall analysis available."
                f.write(analysis)
            except:
                f.write("No overall analysis available.")
        else:
            f.write("No overall analysis available.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unravel AI - Code Graph Analyzer")
    parser.add_argument("--input", "-i", required=True, help="Directory containing decompiled files")
    parser.add_argument("--output", "-o", required=True, help="Directory to save analysis results")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enhancement")
    
    args = parser.parse_args()
    
    analyze_codebase(args.input, args.output, not args.no_llm)