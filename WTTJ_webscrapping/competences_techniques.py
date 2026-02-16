"""
Liste complète des compétences techniques
Utilisable pour la détection de compétences dans les offres d'emploi
"""

COMPETENCES_TECHNIQUES = {
    "langages_programmation": [
        "JavaScript", "TypeScript", "HTML", "HTML5", "CSS", "CSS3", 
        "SCSS", "SASS", "LESS", "PHP", "Ruby", "Python",
        "Java", "Kotlin", "Swift", "Objective-C", "Dart", "Go", "Golang",
        "Rust", "C", "C++", "C#", "Scala", "Perl", "Elixir", "Clojure",
        "R", "Julia", "MATLAB", "VBA",
        "Bash", "Shell", "PowerShell",
        "SQL", "MySQL", "PostgreSQL", "T-SQL", "PL/SQL", "NoSQL", 
        "GraphQL", "SPARQL"
    ],
    
    "frameworks_frontend": [
        "React", "React.js", "ReactJS", "Next.js", "Nextjs",
        "Vue", "Vue.js", "Vuejs", "Nuxt", "Nuxt.js",
        "Angular", "AngularJS", "Svelte", "Ember", "Backbone",
        "jQuery", "Redux", "MobX", "Recoil", "Zustand",
        "Webpack", "Vite", "Parcel", "Rollup", "esbuild",
        "Tailwind", "TailwindCSS", "Bootstrap", "Material-UI", "MUI",
        "Chakra UI", "Ant Design", "Styled Components", "Emotion"
    ],
    
    "frameworks_backend": [
        "Node.js", "Express", "Express.js", "NestJS", "Fastify", "Koa",
        "Django", "Flask", "FastAPI", "Tornado",
        "Ruby on Rails", "Rails", "Sinatra",
        "Spring", "Spring Boot", "Hibernate", "Micronaut", "Quarkus",
        "Laravel", "Symfony", "CodeIgniter", "CakePHP", "Zend", "Slim",
        "ASP.NET", ".NET", ".NET Core", "Entity Framework",
        "Gin", "Echo", "Fiber", "Phoenix", "Ktor"
    ],
    
    "frameworks_mobile": [
        "React Native", "Flutter", "Ionic", "Cordova", "Capacitor",
        "Xamarin", "SwiftUI", "Jetpack Compose", "Kotlin Multiplatform"
    ],
    
    "testing": [
        "Jest", "Mocha", "Chai", "Jasmine", "Karma", "Vitest",
        "Cypress", "Playwright", "Selenium", "Puppeteer", "WebdriverIO",
        "Testing Library", "Enzyme",
        "JUnit", "TestNG", "Mockito", "pytest", "unittest", "nose",
        "RSpec", "Minitest", "Cucumber", "Behave",
        "Postman", "SoapUI", "Insomnia", "REST Assured"
    ],
    
    "databases_sql": [
        "MySQL", "PostgreSQL", "MariaDB", "SQLite",
        "Oracle", "Oracle Database", "SQL Server", "Microsoft SQL Server",
        "DB2", "Teradata", "Snowflake", "Amazon Aurora"
    ],
    
    "databases_nosql": [
        "MongoDB", "Cassandra", "CouchDB", "Redis", "Memcached",
        "DynamoDB", "Firebase", "Firestore", "Realtime Database",
        "Neo4j", "ArangoDB", "OrientDB",
        "Elasticsearch", "Solr", "OpenSearch"
    ],
    
    "devops_containerization": [
        "Docker", "Kubernetes", "K8s", "OpenShift", "Rancher",
        "Podman", "Helm", "Kustomize", "Skaffold", "Tilt"
    ],
    
    "devops_cicd": [
        "Jenkins", "GitLab CI", "GitLab CI/CD", "GitHub Actions",
        "CircleCI", "Travis CI", "Azure DevOps", "Azure Pipelines",
        "Bamboo", "TeamCity", "ArgoCD", "Flux", "Spinnaker",
        "Drone", "Concourse", "GoCD"
    ],
    
    "cloud_aws": [
        "AWS", "Amazon Web Services", "EC2", "S3", "Lambda",
        "ECS", "EKS", "Fargate", "RDS", "Aurora", "DynamoDB",
        "CloudFormation", "CloudWatch", "Route53", "VPC",
        "IAM", "SQS", "SNS", "API Gateway", "CloudFront",
        "Elastic Beanstalk", "Step Functions", "SageMaker"
    ],
    
    "cloud_azure": [
        "Azure", "Microsoft Azure", "Azure Functions", "Azure DevOps",
        "Azure AD", "Azure Storage", "Azure SQL", "Cosmos DB",
        "AKS", "Azure Kubernetes Service", "App Service"
    ],
    
    "cloud_gcp": [
        "GCP", "Google Cloud Platform", "Google Cloud",
        "Cloud Run", "Cloud Functions", "GKE", "BigQuery",
        "Cloud Storage", "Firestore", "Cloud SQL", "Pub/Sub"
    ],
    
    "cloud_autres": [
        "Heroku", "DigitalOcean", "Vercel", "Netlify",
        "Cloudflare", "Railway", "Render", "Fly.io"
    ],
    
    "iac": [
        "Terraform", "Ansible", "Puppet", "Chef",
        "CloudFormation", "Pulumi", "Vagrant", "SaltStack",
        "Packer", "Bicep"
    ],
    
    "monitoring": [
        "Prometheus", "Grafana", "DataDog", "Datadog",
        "New Relic", "Splunk", "ELK", "Elastic Stack",
        "Elasticsearch", "Logstash", "Kibana",
        "Nagios", "Zabbix", "AppDynamics", "Dynatrace",
        "Sentry", "Jaeger", "Zipkin", "OpenTelemetry"
    ],
    
    "data_engineering": [
        "Apache Spark", "Spark", "Hadoop", "Kafka", "Apache Kafka",
        "Airflow", "Apache Airflow", "Flink", "Apache Flink",
        "Storm", "Databricks", "dbt", "Talend", "Informatica",
        "Apache NiFi", "Prefect", "Dagster", "Luigi"
    ],
    
    "ml_ai": [
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "sklearn",
        "XGBoost", "LightGBM", "CatBoost",
        "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
        "NLTK", "spaCy", "Hugging Face", "Transformers",
        "OpenCV", "YOLO", "BERT", "GPT", "LLM",
        "LangChain", "LlamaIndex", "MLflow", "Kubeflow",
        "Stable Diffusion", "Diffusion Models"
    ],
    
    "big_data": [
        "Hadoop", "Hive", "Pig", "HBase", "Cassandra",
        "Spark", "Presto", "Impala", "Redshift", "BigQuery",
        "Snowflake", "Druid", "ClickHouse"
    ],
    
    "cms_ecommerce": [
        "WordPress", "WooCommerce", "Drupal", "Joomla",
        "Magento", "PrestaShop", "Shopify", "Shopify Plus",
        "Strapi", "Contentful", "Sanity", "Ghost",
        "Webflow", "Wix", "Squarespace"
    ],
    
    "version_control": [
        "Git", "GitHub", "GitLab", "Bitbucket",
        "SVN", "Subversion", "Mercurial", "Perforce"
    ],
    
    "methodologies": [
        "Agile", "Scrum", "Kanban", "Lean", "DevOps", "GitOps",
        "CI/CD", "TDD", "Test-Driven Development",
        "BDD", "Behavior-Driven Development",
        "DDD", "Domain-Driven Development",
        "Microservices", "Event-Driven Architecture",
        "REST", "RESTful", "SOAP", "gRPC", "GraphQL",
        "WebSocket", "WebRTC", "OAuth", "JWT", "SAML",
        "MVC", "MVVM", "Clean Architecture", "Hexagonal Architecture",
        "CQRS", "Event Sourcing"
    ],
    
    "design_ux": [
        "Figma", "Adobe XD", "Sketch", "InVision", "Zeplin",
        "Miro", "Balsamiq", "Adobe Photoshop", "Adobe Illustrator",
        "Adobe Creative Suite", "Canva", "Framer", "Principle"
    ],
    
    "collaboration": [
        "Jira", "Confluence", "Trello", "Asana", "Notion",
        "Monday", "Monday.com", "Slack", "Microsoft Teams",
        "Discord", "Linear", "ClickUp", "Airtable"
    ],
    
    "security": [
        "OWASP", "SSL/TLS", "OAuth", "OAuth2", "SAML", "Kerberos",
        "Penetration Testing", "Pen Testing", "Burp Suite",
        "Metasploit", "Wireshark", "Nmap", "Snort",
        "SIEM", "IDS", "IPS", "WAF", "VPN", "Firewall",
        "Encryption", "PKI", "SOC", "CISSP", "CEH",
        "Zero Trust", "GDPR", "ISO 27001", "SOC 2"
    ],
    
    "erp_crm": [
        "SAP", "SAP ERP", "SAP HANA", "Salesforce", "Salesforce CRM",
        "Microsoft Dynamics", "Dynamics 365", "Oracle ERP",
        "NetSuite", "Odoo", "HubSpot", "Zoho", "Zoho CRM",
        "SugarCRM", "Pipedrive", "Freshworks"
    ],
    
    "blockchain": [
        "Ethereum", "Solidity", "Web3", "Web3.js", "ethers.js",
        "Smart Contracts", "Bitcoin", "Hyperledger",
        "Truffle", "Hardhat", "Polygon", "Binance Smart Chain",
        "IPFS", "NFT", "DeFi"
    ],
    
    "systemes": [
        "Linux", "Unix", "Ubuntu", "Debian", "CentOS", "RHEL",
        "Windows Server", "MacOS",
        "Nginx", "Apache", "Apache HTTP Server", "IIS",
        "Load Balancing", "HAProxy", "CDN", "DNS",
        "TCP/IP", "HTTP", "HTTPS", "SSH", "FTP", "SFTP"
    ],
    
    "business_intelligence": [
        "Power BI", "Tableau", "Qlik", "QlikView", "Qlik Sense",
        "Looker", "Metabase", "Superset", "Apache Superset",
        "Google Data Studio", "Looker Studio",
        "Sisense", "SAP BusinessObjects", "MicroStrategy",
        "Pentaho", "Jaspersoft"
    ],
    
    "automatisation": [
        "Zapier", "Make", "Integromat", "n8n", "Apache Camel",
        "Automate.io", "IFTTT",
        "UiPath", "Blue Prism", "Automation Anywhere", "RPA",
        "Power Automate"
    ],
    
    "game_dev": [
        "Unity", "Unity3D", "Unreal Engine", "Unreal",
        "Godot", "Phaser", "Three.js", "Babylon.js",
        "WebGL", "OpenGL", "DirectX", "Vulkan",
        "Blender", "Maya", "3ds Max", "ZBrush"
    ],
    
    "iot": [
        "Arduino", "Raspberry Pi", "ESP32", "ESP8266",
        "MQTT", "CoAP", "LoRaWAN", "Zigbee",
        "Bluetooth LE", "BLE", "Edge Computing", "5G"
    ],
    
    "autres": [
        "API", "SDK", "CLI", "WebAssembly", "WASM",
        "Progressive Web App", "PWA", "Service Worker",
        "Responsive Design", "Mobile First", "SEO",
        "Accessibility", "A11y", "WCAG", "ARIA",
        "i18n", "Internationalization", "l10n",
        "Performance Optimization", "Web Performance",
        "Load Testing", "JMeter", "Gatling", "Locust", "k6",
        "JSON", "XML", "YAML", "TOML", "Protocol Buffers",
        "Regex", "Regular Expressions"
    ]
}

# Liste plate de toutes les compétences
TOUTES_COMPETENCES = []
for category in COMPETENCES_TECHNIQUES.values():
    TOUTES_COMPETENCES.extend(category)

# Supprimer les doublons et trier
TOUTES_COMPETENCES = sorted(list(set(TOUTES_COMPETENCES)))

# Fonction helper pour détecter les compétences dans un texte
def detecter_competences(texte, insensible_casse=True):
    """
    Détecte les compétences présentes dans un texte
    """
    if not texte:
        return []
    
    if insensible_casse:
        texte_lower = texte.lower()
    else:
        texte_lower = texte
    
    competences_trouvees = set()  # Utiliser un set pour éviter les doublons
    
    for competence in TOUTES_COMPETENCES:
        competence_lower = competence.lower() if insensible_casse else competence
        
        # Chercher le mot complet avec word boundaries
        import re
        pattern = r'\b' + re.escape(competence_lower) + r'\b'
        
        if re.search(pattern, texte_lower, re.IGNORECASE):
            competences_trouvees.add(competence)
        else:
            # Chercher aussi sans word boundary pour les cas comme "React.js" dans "React"
            # Ou "Node" dans "Node.js"
            if competence_lower in texte_lower:
                # Vérifier que ce n'est pas une fausse détection
                # Par exemple, éviter de détecter "go" dans "Django"
                if len(competence_lower) > 2:  # Seulement pour les mots de 3+ caractères
                    competences_trouvees.add(competence)
    
    return sorted(list(competences_trouvees))

if __name__ == "__main__":
    print(f"Total de compétences techniques répertoriées : {len(TOUTES_COMPETENCES)}")
    print(f"\nCatégories : {len(COMPETENCES_TECHNIQUES)}")
    
