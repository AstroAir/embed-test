"""
Workflow Automation Example

This example demonstrates automated workflows for the PDF Vector System:
- Document ingestion workflows
- Automated processing pipelines
- Scheduled operations
- Event-driven processing
- Workflow orchestration

Prerequisites:
- PDF Vector System installed
- Understanding of workflow concepts
- Sample documents for processing

Usage:
    python workflow_automation.py

Expected Output:
    - Automated workflow demonstrations
    - Event-driven processing examples
    - Scheduling and orchestration patterns
    - Error handling in workflows

Learning Objectives:
- Learn workflow automation patterns
- Understand event-driven processing
- See scheduling and orchestration techniques
- Master error handling in workflows
"""

import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import queue
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import (
    print_section, print_subsection, example_context,
    format_duration
)
from utils.sample_data_generator import ensure_sample_data


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(Enum):
    """Types of workflow events."""
    DOCUMENT_ADDED = "document_added"
    DOCUMENT_PROCESSED = "document_processed"
    PROCESSING_FAILED = "processing_failed"
    BATCH_COMPLETED = "batch_completed"
    SYSTEM_ERROR = "system_error"


@dataclass
class WorkflowEvent:
    """Represents a workflow event."""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    workflow_id: Optional[str] = None


@dataclass
class WorkflowTask:
    """Represents a task in a workflow."""
    task_id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None


@dataclass
class Workflow:
    """Represents a complete workflow."""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WorkflowEngine:
    """Simple workflow execution engine."""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.event_queue = queue.Queue()
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.running = False
        self.worker_thread = None
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def emit_event(self, event: WorkflowEvent):
        """Emit an event to the queue."""
        self.event_queue.put(event)
    
    def start(self):
        """Start the workflow engine."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._event_loop)
        self.worker_thread.start()
        print("🚀 Workflow engine started")
    
    def stop(self):
        """Stop the workflow engine."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        print("🛑 Workflow engine stopped")
    
    def _event_loop(self):
        """Main event processing loop."""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                self._handle_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in event loop: {e}")
    
    def _handle_event(self, event: WorkflowEvent):
        """Handle a workflow event."""
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"❌ Error in event handler: {e}")
    
    def submit_workflow(self, workflow: Workflow):
        """Submit a workflow for execution."""
        self.workflows[workflow.workflow_id] = workflow
        self._execute_workflow(workflow)
    
    def _execute_workflow(self, workflow: Workflow):
        """Execute a workflow."""
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        
        print(f"🔄 Executing workflow: {workflow.name}")
        
        try:
            # Execute tasks in dependency order
            completed_tasks = set()
            
            while len(completed_tasks) < len(workflow.tasks):
                ready_tasks = [
                    task for task in workflow.tasks
                    if (task.status == WorkflowStatus.PENDING and
                        all(dep in completed_tasks for dep in task.dependencies))
                ]
                
                if not ready_tasks:
                    # Check for failed dependencies
                    failed_tasks = [t for t in workflow.tasks if t.status == WorkflowStatus.FAILED]
                    if failed_tasks:
                        workflow.status = WorkflowStatus.FAILED
                        break
                    else:
                        time.sleep(0.1)  # Wait for tasks to complete
                        continue
                
                for task in ready_tasks:
                    self._execute_task(task, workflow)
                    if task.status == WorkflowStatus.COMPLETED:
                        completed_tasks.add(task.task_id)
                    elif task.status == WorkflowStatus.FAILED:
                        workflow.status = WorkflowStatus.FAILED
                        break
            
            if workflow.status != WorkflowStatus.FAILED:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.now()
                
                # Emit completion event
                self.emit_event(WorkflowEvent(
                    event_type=EventType.BATCH_COMPLETED,
                    timestamp=datetime.now(),
                    data={"workflow_id": workflow.workflow_id},
                    workflow_id=workflow.workflow_id
                ))
        
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            print(f"❌ Workflow failed: {e}")
    
    def _execute_task(self, task: WorkflowTask, workflow: Workflow):
        """Execute a single task."""
        task.status = WorkflowStatus.RUNNING
        task.start_time = datetime.now()
        
        try:
            print(f"   🔧 Executing task: {task.name}")
            task.result = task.function(*task.args, **task.kwargs)
            task.status = WorkflowStatus.COMPLETED
            task.end_time = datetime.now()
            
            duration = task.end_time - task.start_time
            print(f"   ✅ Task completed: {task.name} ({format_duration(duration.total_seconds())})")
            
        except Exception as e:
            task.status = WorkflowStatus.FAILED
            task.error = str(e)
            task.end_time = datetime.now()
            print(f"   ❌ Task failed: {task.name} - {e}")


def setup_pipeline() -> PDFVectorPipeline:
    """Set up the PDF processing pipeline."""
    config = Config()
    config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    config.embedding.model_name = "all-MiniLM-L6-v2"
    config.embedding.batch_size = 16
    config.chroma_db.collection_name = "workflow_automation"
    config.chroma_db.persist_directory = Path("./workflow_automation_db")
    config.debug = False  # Quiet for workflow execution
    
    return PDFVectorPipeline(config)


def create_document_ingestion_workflow(pipeline: PDFVectorPipeline, documents: List[Path]) -> Workflow:
    """Create a document ingestion workflow."""
    
    def validate_documents(docs: List[Path]) -> List[Path]:
        """Validate that documents exist and are readable."""
        valid_docs = []
        for doc in docs:
            if doc.exists() and doc.suffix.lower() == '.pdf':
                valid_docs.append(doc)
            else:
                print(f"   ⚠️  Skipping invalid document: {doc}")
        return valid_docs
    
    def process_document(doc_path: Path) -> Dict[str, Any]:
        """Process a single document."""
        result = pipeline.process_pdf(
            pdf_path=doc_path,
            document_id=f"workflow_{doc_path.stem}",
            show_progress=False
        )
        
        return {
            "document_id": f"workflow_{doc_path.stem}",
            "success": result.success,
            "chunks_processed": result.chunks_processed if result.success else 0,
            "error": result.error_message if not result.success else None
        }
    
    def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of processing results."""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        return {
            "total_documents": len(results),
            "successful_documents": len(successful),
            "failed_documents": len(failed),
            "total_chunks": sum(r["chunks_processed"] for r in successful)
        }
    
    # Create workflow tasks
    tasks = [
        WorkflowTask(
            task_id="validate",
            name="Validate Documents",
            function=validate_documents,
            args=(documents,)
        )
    ]
    
    # Add processing tasks for each document
    for i, doc in enumerate(documents):
        tasks.append(WorkflowTask(
            task_id=f"process_{i}",
            name=f"Process {doc.name}",
            function=process_document,
            args=(doc,),
            dependencies=["validate"]
        ))
    
    # Add summary task
    tasks.append(WorkflowTask(
        task_id="summarize",
        name="Generate Summary",
        function=generate_summary,
        args=([],),  # Will be populated with results
        dependencies=[f"process_{i}" for i in range(len(documents))]
    ))
    
    return Workflow(
        workflow_id=f"ingestion_{int(time.time())}",
        name="Document Ingestion Workflow",
        description=f"Process {len(documents)} documents",
        tasks=tasks
    )


def demonstrate_basic_workflow(engine: WorkflowEngine, pipeline: PDFVectorPipeline) -> None:
    """Demonstrate a basic document processing workflow."""
    print_subsection("Basic Document Processing Workflow")
    
    # Ensure sample data exists
    sample_dir = Path("examples/sample_data")
    if not ensure_sample_data(sample_dir):
        print("⚠️  No sample data available")
        return
    
    # Get available documents
    pdf_files = list(sample_dir.glob("*.pdf"))[:2]  # Limit to 2 for demo
    
    if not pdf_files:
        print("📄 No PDF files found")
        return
    
    print(f"📚 Creating workflow for {len(pdf_files)} documents:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file.name}")
    
    # Create and submit workflow
    workflow = create_document_ingestion_workflow(pipeline, pdf_files)
    engine.submit_workflow(workflow)
    
    # Wait for completion
    start_time = time.time()
    timeout = 60  # 60 seconds timeout
    
    while workflow.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
        if time.time() - start_time > timeout:
            print("⏰ Workflow timeout")
            break
        time.sleep(1)
    
    # Report results
    if workflow.status == WorkflowStatus.COMPLETED:
        duration = workflow.completed_at - workflow.started_at
        print(f"✅ Workflow completed in {format_duration(duration.total_seconds())}")
        
        # Show task results
        for task in workflow.tasks:
            if task.status == WorkflowStatus.COMPLETED:
                print(f"   ✅ {task.name}")
            else:
                print(f"   ❌ {task.name}: {task.error}")
    else:
        print(f"❌ Workflow failed with status: {workflow.status.value}")


def demonstrate_event_driven_processing(engine: WorkflowEngine) -> None:
    """Demonstrate event-driven workflow processing."""
    print_subsection("Event-Driven Processing")
    
    # Event handler for document processing
    def on_document_processed(event: WorkflowEvent):
        """Handle document processed event."""
        data = event.data
        print(f"   📄 Document processed: {data.get('document_id', 'Unknown')}")
        
        if data.get('success', False):
            print(f"      ✅ Success: {data.get('chunks_processed', 0)} chunks")
        else:
            print(f"      ❌ Failed: {data.get('error', 'Unknown error')}")
    
    # Event handler for batch completion
    def on_batch_completed(event: WorkflowEvent):
        """Handle batch completion event."""
        print(f"   🎉 Batch completed for workflow: {event.workflow_id}")
        print(f"      Timestamp: {event.timestamp}")
    
    # Event handler for system errors
    def on_system_error(event: WorkflowEvent):
        """Handle system error event."""
        print(f"   🚨 System error: {event.data.get('error', 'Unknown error')}")
        print(f"      Workflow: {event.workflow_id}")
    
    # Register event handlers
    engine.register_event_handler(EventType.DOCUMENT_PROCESSED, on_document_processed)
    engine.register_event_handler(EventType.BATCH_COMPLETED, on_batch_completed)
    engine.register_event_handler(EventType.SYSTEM_ERROR, on_system_error)
    
    print("📡 Event handlers registered:")
    print("   - Document processed events")
    print("   - Batch completion events")
    print("   - System error events")
    
    # Simulate some events
    print("\n🎭 Simulating events:")
    
    # Document processed event
    engine.emit_event(WorkflowEvent(
        event_type=EventType.DOCUMENT_PROCESSED,
        timestamp=datetime.now(),
        data={
            "document_id": "sample_doc_1",
            "success": True,
            "chunks_processed": 25
        },
        workflow_id="demo_workflow"
    ))
    
    # Processing failed event
    engine.emit_event(WorkflowEvent(
        event_type=EventType.PROCESSING_FAILED,
        timestamp=datetime.now(),
        data={
            "document_id": "sample_doc_2",
            "error": "File not found"
        },
        workflow_id="demo_workflow"
    ))
    
    # Batch completed event
    engine.emit_event(WorkflowEvent(
        event_type=EventType.BATCH_COMPLETED,
        timestamp=datetime.now(),
        data={"total_documents": 2, "successful": 1, "failed": 1},
        workflow_id="demo_workflow"
    ))
    
    # Give events time to process
    time.sleep(2)


def demonstrate_scheduled_workflows(engine: WorkflowEngine) -> None:
    """Demonstrate scheduled workflow execution."""
    print_subsection("Scheduled Workflows")
    
    print("⏰ Scheduled workflow patterns:")
    
    # Simulate different scheduling scenarios
    schedules = [
        {
            "name": "Hourly Document Scan",
            "description": "Scan for new documents every hour",
            "frequency": "hourly",
            "next_run": datetime.now() + timedelta(hours=1)
        },
        {
            "name": "Daily Index Optimization",
            "description": "Optimize search index daily at 2 AM",
            "frequency": "daily",
            "next_run": datetime.now().replace(hour=2, minute=0, second=0) + timedelta(days=1)
        },
        {
            "name": "Weekly Backup",
            "description": "Backup database weekly on Sundays",
            "frequency": "weekly",
            "next_run": datetime.now() + timedelta(days=7)
        }
    ]
    
    for schedule in schedules:
        print(f"\n📅 {schedule['name']}:")
        print(f"   Description: {schedule['description']}")
        print(f"   Frequency: {schedule['frequency']}")
        print(f"   Next run: {schedule['next_run'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate time until next run
        time_until = schedule['next_run'] - datetime.now()
        if time_until.total_seconds() > 0:
            print(f"   Time until next run: {format_duration(time_until.total_seconds())}")
        else:
            print(f"   ⚠️  Overdue by: {format_duration(-time_until.total_seconds())}")


def main() -> None:
    """
    Demonstrate workflow automation for the PDF Vector System.
    
    This function shows how to create automated workflows,
    handle events, and orchestrate complex processing tasks.
    """
    with example_context("Workflow Automation"):
        
        print_section("Workflow Automation Setup")
        
        # Set up components
        print_subsection("Initializing Components")
        
        pipeline = setup_pipeline()
        print("✅ PDF processing pipeline initialized")
        
        engine = WorkflowEngine()
        engine.start()
        
        try:
            print_section("Workflow Demonstrations")
            
            # Demonstrate different workflow patterns
            demonstrate_basic_workflow(engine, pipeline)
            demonstrate_event_driven_processing(engine)
            demonstrate_scheduled_workflows(engine)
            
            print_section("Workflow Automation Summary")
            
            print("🎯 Workflow Automation Best Practices:")
            print()
            print("1. 🔄 Workflow Design:")
            print("   - Break complex processes into discrete tasks")
            print("   - Define clear dependencies between tasks")
            print("   - Implement proper error handling and recovery")
            print()
            print("2. 📡 Event-Driven Architecture:")
            print("   - Use events to decouple workflow components")
            print("   - Implement event handlers for different scenarios")
            print("   - Log events for debugging and monitoring")
            print()
            print("3. ⏰ Scheduling:")
            print("   - Use appropriate scheduling for different tasks")
            print("   - Implement retry mechanisms for failed tasks")
            print("   - Monitor scheduled tasks for failures")
            print()
            print("4. 🛡️  Error Handling:")
            print("   - Implement graceful degradation")
            print("   - Use circuit breakers for external dependencies")
            print("   - Provide clear error messages and logging")
            print()
            print("5. 📊 Monitoring:")
            print("   - Track workflow execution metrics")
            print("   - Implement alerting for failures")
            print("   - Provide visibility into workflow status")
            print()
            print("Next steps:")
            print("- Implement your own workflow patterns")
            print("- Explore production deployment in 10_production/")
            print("- Try the monitoring examples for workflow observability")
        
        finally:
            # Clean up
            engine.stop()


if __name__ == "__main__":
    main()
