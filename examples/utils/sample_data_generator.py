"""
Sample data generator for PDF Vector System examples.

This module creates sample PDF files and test text content for use in
examples when real data is not available.

Prerequisites:
- VectorFlow examples environment
- Optional: PyMuPDF (fitz) installed to generate real PDF files

Usage:
    from examples.utils.sample_data_generator import (
        create_sample_pdf,
        get_sample_content,
        ensure_sample_data,
    )

Expected Output:
    - When used by examples, creates one or more sample PDFs in
      "examples/sample_data" if real documents are missing
    - Provides realistic sample content for different document types

Learning Objectives:
- Understand how sample documents for the examples are generated
- See how to build reusable utilities for test data creation
- Learn how to fall back to synthetic data when real PDFs are not available
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def create_sample_pdf(
    output_path: Path,
    title: str,
    content: str,
    author: str = "PDF Vector System",
    subject: str = "Sample Document",
) -> bool:
    """
    Create a sample PDF file with the given content.

    Args:
        output_path: Path where to save the PDF
        title: Document title
        content: Text content for the PDF
        author: Document author
        subject: Document subject

    Returns:
        True if PDF was created successfully, False otherwise
    """
    if not PYMUPDF_AVAILABLE:
        return False

    try:
        # Create a new PDF document
        doc = fitz.open()

        # Add metadata
        metadata = {
            "title": title,
            "author": author,
            "subject": subject,
            "creator": "PDF Vector System Sample Generator",
            "producer": "PyMuPDF",
            "creationDate": datetime.now(),
            "modDate": datetime.now(),
        }
        doc.set_metadata(metadata)

        # Split content into pages (roughly 3000 characters per page)
        page_size = 3000
        pages = [content[i : i + page_size] for i in range(0, len(content), page_size)]

        for page_content in pages:
            # Create a new page
            page = doc.new_page()

            # Define text rectangle (with margins)
            rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)

            # Insert text
            page.insert_textbox(
                rect,
                page_content,
                fontsize=11,
                fontname="helv",
                color=(0, 0, 0),
                align=0,
            )

        # Save the PDF
        doc.save(str(output_path))
        doc.close()

        return True

    except Exception:
        return False


def get_sample_content(content_type: str) -> tuple[str, str]:
    """
    Get sample content for different document types.

    Args:
        content_type: Type of content (research, technical, business, legal)

    Returns:
        Tuple of (title, content)
    """
    content_library = {
        "research": (
            "Machine Learning in Natural Language Processing: A Comprehensive Survey",
            """
            Abstract

            Natural Language Processing (NLP) has experienced remarkable advances with the integration of machine learning techniques. This comprehensive survey examines the evolution of machine learning approaches in NLP, from traditional statistical methods to modern deep learning architectures.

            1. Introduction

            Natural Language Processing represents one of the most challenging domains in artificial intelligence, requiring systems to understand, interpret, and generate human language. The field has undergone significant transformation with the advent of machine learning, particularly deep learning methodologies.

            The complexity of human language, with its inherent ambiguity, context-dependency, and cultural nuances, presents unique challenges that traditional rule-based systems struggled to address effectively. Machine learning approaches have provided more robust and scalable solutions to these challenges.

            2. Historical Perspective

            The journey of machine learning in NLP began with statistical approaches in the 1990s. Early methods relied heavily on n-gram models and hidden Markov models for tasks such as part-of-speech tagging and named entity recognition.

            The introduction of support vector machines and maximum entropy models marked a significant advancement, providing better generalization capabilities and handling of sparse feature spaces common in NLP tasks.

            3. Deep Learning Revolution

            The emergence of deep learning has revolutionized NLP, beginning with the introduction of word embeddings such as Word2Vec and GloVe. These dense vector representations captured semantic relationships between words in ways that traditional one-hot encodings could not.

            Recurrent Neural Networks (RNNs) and their variants, including Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs), enabled the modeling of sequential dependencies in text, leading to significant improvements in tasks such as machine translation and sentiment analysis.

            4. Transformer Architecture

            The introduction of the Transformer architecture marked a paradigm shift in NLP. The attention mechanism allowed models to capture long-range dependencies more effectively than RNNs, while enabling parallel processing that significantly reduced training time.

            BERT (Bidirectional Encoder Representations from Transformers) demonstrated the power of pre-training on large corpora followed by fine-tuning on specific tasks. This approach achieved state-of-the-art results across numerous NLP benchmarks.

            5. Large Language Models

            The development of increasingly large language models, such as GPT-3 and its successors, has pushed the boundaries of what is possible in NLP. These models demonstrate emergent capabilities, including few-shot learning and reasoning abilities that were previously thought to require explicit programming.

            The scale of these models, with billions or even trillions of parameters, has raised important questions about computational efficiency, environmental impact, and accessibility of advanced NLP capabilities.

            6. Current Challenges and Future Directions

            Despite remarkable progress, several challenges remain in machine learning for NLP. These include handling low-resource languages, improving model interpretability, addressing bias and fairness concerns, and developing more efficient architectures.

            Future research directions include multimodal learning, incorporating external knowledge, and developing more sample-efficient learning algorithms. The integration of symbolic reasoning with neural approaches also presents promising opportunities.

            7. Conclusion

            Machine learning has fundamentally transformed natural language processing, enabling capabilities that seemed impossible just a few decades ago. As we continue to push the boundaries of what is possible, it is crucial to address the challenges and ethical considerations that accompany these powerful technologies.

            The future of NLP lies in developing more efficient, interpretable, and fair systems that can truly understand and assist humans in their communication and information processing needs.
            """,
        ),
        "technical": (
            "Microservices Architecture: Design Patterns and Best Practices",
            """
            Table of Contents

            1. Introduction to Microservices
            2. Architectural Principles
            3. Design Patterns
            4. Communication Strategies
            5. Data Management
            6. Deployment and Operations
            7. Monitoring and Observability
            8. Security Considerations

            1. Introduction to Microservices

            Microservices architecture has emerged as a dominant pattern for building scalable, maintainable, and resilient distributed systems. This architectural style structures an application as a collection of loosely coupled services, each responsible for a specific business capability.

            The microservices approach contrasts with traditional monolithic architectures by decomposing applications into smaller, independent services that can be developed, deployed, and scaled independently. This decomposition enables organizations to achieve greater agility, scalability, and technological diversity.

            Key characteristics of microservices include:
            - Single responsibility principle
            - Decentralized governance
            - Failure isolation
            - Technology diversity
            - Independent deployment

            2. Architectural Principles

            Successful microservices implementations adhere to several core principles that guide design decisions and operational practices.

            2.1 Service Autonomy
            Each microservice should be autonomous, owning its data and business logic. This autonomy enables teams to make independent decisions about technology choices, deployment schedules, and scaling strategies.

            2.2 Decentralized Data Management
            Rather than sharing databases, each microservice should manage its own data store. This approach eliminates tight coupling between services and enables each service to choose the most appropriate data storage technology for its specific needs.

            2.3 Design for Failure
            In distributed systems, failures are inevitable. Microservices architectures must be designed with failure in mind, implementing patterns such as circuit breakers, bulkheads, and timeouts to prevent cascading failures.

            3. Design Patterns

            Several design patterns have emerged as best practices for microservices architectures.

            3.1 API Gateway Pattern
            An API Gateway serves as a single entry point for client requests, providing cross-cutting concerns such as authentication, rate limiting, and request routing. This pattern simplifies client interactions and centralizes common functionality.

            3.2 Circuit Breaker Pattern
            The Circuit Breaker pattern prevents cascading failures by monitoring service calls and temporarily blocking requests to failing services. This pattern improves system resilience and provides graceful degradation.

            3.3 Saga Pattern
            For managing distributed transactions across multiple services, the Saga pattern provides a way to maintain data consistency without traditional ACID transactions. Sagas coordinate a series of local transactions, with compensation actions to handle failures.

            4. Communication Strategies

            Effective communication between microservices is crucial for system functionality and performance.

            4.1 Synchronous Communication
            REST APIs and GraphQL provide synchronous communication mechanisms suitable for request-response interactions. These approaches offer simplicity and immediate consistency but can create tight coupling and availability dependencies.

            4.2 Asynchronous Communication
            Message queues and event streaming platforms enable asynchronous communication, improving system resilience and scalability. Event-driven architectures promote loose coupling and enable reactive system behaviors.

            5. Data Management

            Data management in microservices presents unique challenges and opportunities.

            5.1 Database per Service
            Each microservice should own its data, choosing the most appropriate database technology for its specific requirements. This approach enables polyglot persistence but requires careful consideration of data consistency and integration.

            5.2 Event Sourcing
            Event sourcing captures all changes to application state as a sequence of events. This approach provides complete audit trails and enables temporal queries, but requires careful design of event schemas and handling of event evolution.

            6. Deployment and Operations

            Microservices deployment and operations require sophisticated tooling and practices.

            6.1 Containerization
            Containers provide consistent deployment environments and resource isolation for microservices. Container orchestration platforms like Kubernetes automate deployment, scaling, and management of containerized services.

            6.2 Continuous Integration and Deployment
            CI/CD pipelines enable rapid, reliable deployment of microservices. Automated testing, including contract testing and end-to-end testing, ensures system quality and reliability.

            7. Monitoring and Observability

            Comprehensive monitoring and observability are essential for operating microservices in production.

            7.1 Distributed Tracing
            Distributed tracing tracks requests across multiple services, providing visibility into system behavior and performance bottlenecks. Tools like Jaeger and Zipkin enable detailed analysis of request flows.

            7.2 Metrics and Logging
            Centralized logging and metrics collection provide insights into system health and performance. Structured logging and correlation IDs enable effective troubleshooting and analysis.

            8. Security Considerations

            Security in microservices architectures requires a comprehensive approach addressing multiple layers and concerns.

            8.1 Zero Trust Architecture
            Zero trust principles assume no implicit trust within the system, requiring authentication and authorization for all service interactions. This approach improves security posture but requires careful implementation of identity and access management.

            8.2 Secrets Management
            Secure handling of secrets, including API keys and database credentials, is crucial for microservices security. Dedicated secrets management systems provide secure storage and rotation of sensitive information.

            Conclusion

            Microservices architecture offers significant benefits for building scalable, maintainable systems, but requires careful consideration of design patterns, operational practices, and organizational factors. Success with microservices depends on adopting appropriate patterns, investing in tooling and automation, and building organizational capabilities to support distributed system development and operations.
            """,
        ),
        "business": (
            "Digital Transformation Strategy: A Comprehensive Business Analysis",
            """
            Executive Summary

            This report presents a comprehensive analysis of digital transformation strategies and their impact on modern business operations. Our research indicates that organizations implementing comprehensive digital transformation initiatives achieve 23% higher revenue growth and 12% greater operational efficiency compared to their traditional counterparts.

            The digital transformation landscape continues to evolve rapidly, driven by technological advances, changing customer expectations, and competitive pressures. Organizations that fail to adapt risk losing market share and relevance in an increasingly digital economy.

            Key Findings:
            - 78% of organizations consider digital transformation a top strategic priority
            - Companies with mature digital capabilities outperform peers by 26% in profitability
            - Customer experience improvements drive 67% of digital transformation initiatives
            - Cloud adoption has accelerated by 300% in the past two years

            1. Introduction

            Digital transformation represents a fundamental shift in how organizations operate, deliver value to customers, and compete in the marketplace. This transformation goes beyond simple technology adoption to encompass changes in business models, organizational culture, and operational processes.

            The COVID-19 pandemic has accelerated digital transformation timelines, with many organizations compressing multi-year initiatives into months. This acceleration has highlighted both the opportunities and challenges associated with rapid digital change.

            Our analysis examines successful digital transformation strategies across multiple industries, identifying common patterns, best practices, and critical success factors.

            2. Market Analysis

            2.1 Current Market Conditions
            The global digital transformation market is projected to reach $1.8 trillion by 2025, representing a compound annual growth rate of 16.5%. This growth is driven by several key factors:

            - Increasing customer expectations for digital experiences
            - Competitive pressure from digital-native companies
            - Operational efficiency requirements
            - Regulatory compliance and reporting needs
            - Remote work and collaboration requirements

            2.2 Industry Trends
            Several trends are shaping the digital transformation landscape:

            Cloud-First Strategies: Organizations are adopting cloud-first approaches to infrastructure and application development, enabling greater scalability and flexibility.

            Data-Driven Decision Making: Advanced analytics and artificial intelligence are becoming central to business operations, enabling more informed and timely decision-making.

            Customer Experience Focus: Digital transformation initiatives increasingly prioritize customer experience improvements, recognizing the direct correlation between customer satisfaction and business performance.

            Ecosystem Integration: Organizations are building integrated digital ecosystems that connect customers, partners, and suppliers through digital platforms and APIs.

            3. Strategic Framework

            3.1 Digital Maturity Assessment
            Successful digital transformation begins with an honest assessment of current digital maturity. Our framework evaluates organizations across five dimensions:

            Technology Infrastructure: The foundation of digital capabilities, including cloud adoption, data architecture, and integration capabilities.

            Digital Processes: The extent to which business processes are digitized and automated.

            Data and Analytics: The organization's ability to collect, analyze, and act on data insights.

            Customer Experience: The quality and consistency of digital customer touchpoints.

            Organizational Culture: The organization's readiness for digital change, including leadership commitment and employee digital skills.

            3.2 Transformation Roadmap
            Based on maturity assessment results, organizations should develop a phased transformation roadmap that addresses immediate needs while building toward long-term objectives.

            Phase 1: Foundation Building (6-12 months)
            - Cloud infrastructure migration
            - Data governance implementation
            - Basic process digitization
            - Employee training and change management

            Phase 2: Capability Development (12-18 months)
            - Advanced analytics implementation
            - Customer experience optimization
            - Process automation expansion
            - Digital product development

            Phase 3: Innovation and Optimization (18+ months)
            - Artificial intelligence integration
            - Ecosystem platform development
            - Continuous innovation processes
            - Advanced personalization capabilities

            4. Implementation Best Practices

            4.1 Leadership and Governance
            Successful digital transformation requires strong leadership commitment and clear governance structures. Key elements include:

            - Executive sponsorship and accountability
            - Cross-functional transformation teams
            - Clear success metrics and KPIs
            - Regular progress reviews and course corrections

            4.2 Technology Strategy
            Technology choices should align with business objectives and support long-term scalability:

            - Cloud-native architecture design
            - API-first integration approach
            - Microservices and containerization
            - DevOps and continuous delivery practices

            4.3 Change Management
            Organizational change management is critical for transformation success:

            - Comprehensive communication strategies
            - Employee training and reskilling programs
            - Cultural change initiatives
            - Resistance management and support systems

            5. Financial Analysis

            5.1 Investment Requirements
            Digital transformation requires significant upfront investment, with typical organizations investing 3-5% of annual revenue in transformation initiatives. Investment areas include:

            - Technology infrastructure and platforms (40%)
            - Software and applications (25%)
            - Training and change management (20%)
            - External consulting and services (15%)

            5.2 Return on Investment
            Organizations typically see positive ROI within 18-24 months of transformation initiation. Key value drivers include:

            - Operational efficiency improvements (30-40% cost reduction)
            - Revenue growth from new digital products and services (15-25% increase)
            - Customer experience improvements (20-30% increase in satisfaction)
            - Risk reduction and compliance benefits (10-15% cost avoidance)

            6. Risk Management

            Digital transformation initiatives face several categories of risk that must be actively managed:

            6.1 Technology Risks
            - Cybersecurity vulnerabilities
            - System integration challenges
            - Technology obsolescence
            - Vendor dependency

            6.2 Organizational Risks
            - Change resistance
            - Skill gaps and talent shortages
            - Cultural misalignment
            - Leadership turnover

            6.3 Market Risks
            - Competitive response
            - Customer adoption challenges
            - Regulatory changes
            - Economic uncertainty

            7. Success Metrics and KPIs

            Measuring digital transformation success requires a balanced scorecard approach incorporating multiple perspectives:

            Financial Metrics:
            - Revenue growth from digital channels
            - Cost reduction from automation
            - Return on digital investment
            - Time to market for new products

            Customer Metrics:
            - Digital customer satisfaction scores
            - Digital channel adoption rates
            - Customer lifetime value
            - Net promoter scores

            Operational Metrics:
            - Process automation rates
            - System uptime and reliability
            - Data quality and accessibility
            - Employee productivity measures

            Innovation Metrics:
            - Number of digital initiatives launched
            - Speed of innovation cycles
            - Digital capability maturity scores
            - Technology adoption rates

            8. Future Outlook

            The digital transformation landscape will continue to evolve, driven by emerging technologies and changing business requirements. Key trends to watch include:

            - Artificial intelligence and machine learning integration
            - Internet of Things (IoT) and edge computing adoption
            - Blockchain and distributed ledger technologies
            - Augmented and virtual reality applications
            - Quantum computing potential

            Organizations that establish strong digital foundations today will be better positioned to capitalize on future technological advances and market opportunities.

            Conclusion

            Digital transformation is no longer optional for organizations seeking to remain competitive in the modern economy. Success requires a comprehensive approach that addresses technology, processes, people, and culture. Organizations that commit to systematic transformation efforts, supported by strong leadership and adequate investment, can achieve significant competitive advantages and sustainable growth.

            The key to success lies in viewing digital transformation not as a one-time project, but as an ongoing capability that enables continuous adaptation and innovation in response to changing market conditions and customer needs.
            """,
        ),
        "legal": (
            "Data Privacy and Protection: Legal Framework Analysis",
            """
            Legal Analysis: Data Privacy and Protection Regulations

            Table of Contents
            1. Introduction
            2. Regulatory Landscape Overview
            3. GDPR Compliance Requirements
            4. CCPA and State-Level Regulations
            5. International Frameworks
            6. Compliance Strategies
            7. Risk Assessment and Mitigation
            8. Conclusion and Recommendations

            1. Introduction

            The legal landscape surrounding data privacy and protection has undergone significant transformation in recent years. Organizations operating in multiple jurisdictions face an increasingly complex web of regulations, each with specific requirements for data collection, processing, storage, and transfer.

            This analysis examines the current regulatory environment, focusing on major frameworks such as the General Data Protection Regulation (GDPR), California Consumer Privacy Act (CCPA), and emerging international standards. We provide practical guidance for organizations seeking to achieve and maintain compliance across multiple jurisdictions.

            The stakes for non-compliance continue to rise, with regulatory authorities imposing substantial fines and enforcement actions. Organizations must adopt comprehensive privacy programs that address both current requirements and anticipated regulatory developments.

            2. Regulatory Landscape Overview

            2.1 Global Trends in Privacy Regulation

            Privacy regulation has evolved from sector-specific rules to comprehensive frameworks covering all aspects of personal data processing. Key trends include:

            - Extraterritorial application of privacy laws
            - Increased individual rights and control over personal data
            - Mandatory breach notification requirements
            - Significant financial penalties for non-compliance
            - Privacy by design and default requirements

            2.2 Jurisdictional Considerations

            Organizations must navigate varying requirements across jurisdictions:

            European Union: GDPR sets the global standard for comprehensive privacy protection, with broad territorial scope and substantial penalties.

            United States: Sectoral approach with federal laws covering specific industries, supplemented by comprehensive state-level regulations like CCPA.

            Asia-Pacific: Emerging frameworks in countries like Singapore, Australia, and Japan, often influenced by GDPR principles.

            Latin America: Growing adoption of comprehensive privacy laws, with Brazil's LGPD leading regional developments.

            3. GDPR Compliance Requirements

            3.1 Scope and Applicability

            The GDPR applies to organizations that:
            - Are established in the EU and process personal data
            - Offer goods or services to EU data subjects
            - Monitor behavior of EU data subjects

            The regulation covers all personal data processing, regardless of the technology used or the location of processing activities.

            3.2 Key Principles

            GDPR establishes six fundamental principles for data processing:

            Lawfulness, Fairness, and Transparency: Processing must have a legal basis and be conducted fairly with clear information provided to data subjects.

            Purpose Limitation: Data must be collected for specified, explicit, and legitimate purposes and not processed incompatibly with those purposes.

            Data Minimization: Processing should be adequate, relevant, and limited to what is necessary for the specified purposes.

            Accuracy: Personal data must be accurate and kept up to date, with inaccurate data erased or rectified without delay.

            Storage Limitation: Data should be kept in identifiable form only as long as necessary for the specified purposes.

            Integrity and Confidentiality: Processing must ensure appropriate security through technical and organizational measures.

            3.3 Individual Rights

            GDPR grants extensive rights to data subjects:

            Right of Access: Individuals can obtain confirmation of processing and access to their personal data.

            Right to Rectification: Individuals can request correction of inaccurate personal data.

            Right to Erasure: Individuals can request deletion of personal data under specific circumstances.

            Right to Restrict Processing: Individuals can limit how their data is processed in certain situations.

            Right to Data Portability: Individuals can receive their data in a structured, machine-readable format.

            Right to Object: Individuals can object to processing based on legitimate interests or for direct marketing.

            3.4 Organizational Obligations

            Organizations must implement comprehensive privacy programs including:

            Data Protection Impact Assessments (DPIAs): Required for high-risk processing activities.

            Privacy by Design and Default: Privacy considerations must be integrated into system design and default settings.

            Records of Processing Activities: Detailed documentation of all processing activities must be maintained.

            Data Protection Officer (DPO): Required for public authorities and organizations engaged in large-scale monitoring or processing of sensitive data.

            Breach Notification: Data breaches must be reported to supervisory authorities within 72 hours and to affected individuals when required.

            4. CCPA and State-Level Regulations

            4.1 California Consumer Privacy Act (CCPA)

            The CCPA grants California residents specific rights regarding their personal information:

            Right to Know: Consumers can request information about data collection and use practices.

            Right to Delete: Consumers can request deletion of personal information.

            Right to Opt-Out: Consumers can opt-out of the sale of personal information.

            Right to Non-Discrimination: Businesses cannot discriminate against consumers for exercising their privacy rights.

            4.2 Emerging State Regulations

            Several states have enacted or are considering comprehensive privacy legislation:

            Virginia Consumer Data Protection Act (VCDPA): Effective January 2023, with similar rights to CCPA but different scope and requirements.

            Colorado Privacy Act (CPA): Effective July 2023, incorporating data protection assessments and universal opt-out mechanisms.

            Connecticut Data Privacy Act: Effective July 2023, with comprehensive consumer rights and business obligations.

            5. International Frameworks

            5.1 Asia-Pacific Developments

            Singapore Personal Data Protection Act (PDPA): Comprehensive framework with mandatory breach notification and data protection officer requirements.

            Australia Privacy Act: Notifiable data breach scheme and ongoing reforms to align with international standards.

            Japan Act on Protection of Personal Information (APPI): Recent amendments strengthening individual rights and cross-border transfer restrictions.

            5.2 Latin American Regulations

            Brazil Lei Geral de Proteção de Dados (LGPD): Comprehensive privacy law modeled on GDPR with similar principles and requirements.

            Argentina Personal Data Protection Law: Established framework with recent updates to address digital developments.

            6. Compliance Strategies

            6.1 Privacy Program Development

            Organizations should establish comprehensive privacy programs incorporating:

            Governance Structure: Clear roles and responsibilities for privacy compliance, including executive oversight and cross-functional coordination.

            Policy Framework: Comprehensive policies covering all aspects of data processing, including collection, use, storage, and transfer.

            Training and Awareness: Regular training programs for employees handling personal data, with specialized training for key roles.

            Vendor Management: Due diligence and contractual protections for third-party data processors and service providers.

            6.2 Technical and Organizational Measures

            Effective compliance requires both technical and organizational safeguards:

            Technical Measures:
            - Encryption of personal data in transit and at rest
            - Access controls and authentication mechanisms
            - Data loss prevention and monitoring systems
            - Automated data retention and deletion processes

            Organizational Measures:
            - Privacy impact assessment procedures
            - Incident response and breach notification processes
            - Regular compliance audits and assessments
            - Documentation and record-keeping systems

            6.3 Cross-Border Data Transfers

            International data transfers require specific safeguards:

            Adequacy Decisions: Transfers to countries with adequate protection as determined by relevant authorities.

            Standard Contractual Clauses: EU-approved contract terms for transfers to countries without adequacy decisions.

            Binding Corporate Rules: Internal policies for multinational organizations governing intra-group transfers.

            Certification Mechanisms: Industry-specific certification programs providing transfer safeguards.

            7. Risk Assessment and Mitigation

            7.1 Compliance Risk Factors

            Organizations face various compliance risks:

            Regulatory Risk: Changes in laws and enforcement priorities requiring program updates.

            Operational Risk: Inadequate processes or controls leading to compliance failures.

            Technology Risk: System vulnerabilities or failures compromising data protection.

            Third-Party Risk: Vendor or partner non-compliance affecting organizational obligations.

            7.2 Risk Mitigation Strategies

            Effective risk mitigation requires:

            Regular Risk Assessments: Systematic evaluation of privacy risks across all business activities.

            Continuous Monitoring: Ongoing surveillance of compliance status and emerging risks.

            Incident Response Planning: Prepared procedures for addressing privacy incidents and breaches.

            Insurance Coverage: Appropriate cyber liability and privacy insurance to address financial risks.

            8. Conclusion and Recommendations

            The data privacy regulatory landscape will continue to evolve, with new laws and enforcement actions shaping organizational obligations. Success requires a proactive approach that anticipates regulatory developments and builds flexible compliance capabilities.

            Key recommendations for organizations include:

            1. Establish comprehensive privacy governance with executive oversight and clear accountability.

            2. Implement privacy by design principles in all system development and business process design.

            3. Develop robust vendor management programs addressing third-party privacy risks.

            4. Invest in privacy-enhancing technologies that support compliance while enabling business objectives.

            5. Maintain current awareness of regulatory developments and enforcement trends across relevant jurisdictions.

            6. Conduct regular compliance assessments and update programs based on identified gaps and emerging requirements.

            Organizations that view privacy compliance as a strategic capability rather than a compliance burden will be better positioned to build customer trust, avoid regulatory penalties, and capitalize on the competitive advantages of strong privacy practices.

            The investment in comprehensive privacy programs pays dividends through reduced regulatory risk, enhanced customer relationships, and improved operational efficiency through better data governance practices.
            """,
        ),
    }

    return content_library.get(content_type, content_library["research"])


def create_sample_pdfs(
    output_dir: Path, num_files: int = 3, content_types: Optional[list[str]] = None
) -> list[Path]:
    """
    Create multiple sample PDF files.

    Args:
        output_dir: Directory where to save the PDFs
        num_files: Number of PDF files to create
        content_types: List of content types to use

    Returns:
        List of created PDF file paths
    """
    if content_types is None:
        content_types = ["research", "technical", "business", "legal"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    for i in range(num_files):
        content_type = content_types[i % len(content_types)]
        title, content = get_sample_content(content_type)

        filename = f"sample_{content_type}_{i + 1}.pdf"
        output_path = output_dir / filename

        if create_sample_pdf(output_path, title, content):
            created_files.append(output_path)

    return created_files


def ensure_sample_data(sample_dir: Path) -> bool:
    """
    Ensure sample data exists, creating it if necessary.

    Args:
        sample_dir: Directory for sample data

    Returns:
        True if sample data is available, False otherwise
    """
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing PDF files
    existing_pdfs = list(sample_dir.glob("*.pdf"))

    if existing_pdfs:
        return True

    # Create sample PDFs if none exist

    if not PYMUPDF_AVAILABLE:
        return False

    created_files = create_sample_pdfs(sample_dir, num_files=3)

    return bool(created_files)


if __name__ == "__main__":
    # Create sample data when run directly
    sample_dir = Path("examples/sample_data")
    ensure_sample_data(sample_dir)
