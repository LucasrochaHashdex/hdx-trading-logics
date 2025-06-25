# 🚀 Trading Process Management System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![BigQuery](https://img.shields.io/badge/Google-BigQuery-orange.svg)](https://cloud.google.com/bigquery)
[![License](https://img.shields.io/badge/License-Private-red.svg)]()

A comprehensive trading workflow management system for automated ETF trading operations, primary unit calculations, and adjustment cycles.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)

## 🎯 Overview

The Trading Process Management System is designed to streamline and automate daily trading operations for ETF funds. It provides a complete workflow from initial order generation to final allocation adjustments, with seamless integration to BigQuery and trading APIs.

### Key Components

- **🌅 Initial Orders**: Generate and organize trading orders
- **🏭 Primary Units**: Calculate optimal primary unit allocations
- **🔄 Adjustments**: Run iterative adjustment cycles
- **📊 Allocation Solver**: Optimize share distributions across funds
- **🔌 API Integration**: Connect with trading platforms and data sources

## ✨ Features

- **Automated Workflow Management**: Complete daily trading workflow automation
- **Smart Allocation**: Advanced optimization algorithms for fund allocations
- **Real-time Adjustments**: Dynamic adjustment cycles based on market execution
- **BigQuery Integration**: Seamless data storage and retrieval
- **Email Notifications**: Automated trading instruction emails
- **Error Handling**: Robust error handling and logging
- **Modular Design**: Clean, maintainable code architecture

## 📁 Project Structure

```
Trading_Process/
├── 📄 README.md                           # This file
├── 📄 requirements.txt                    # Python dependencies
├── 📄 daily_trading_workflow.py           # Main workflow interface
├── src/                                   # Source code directory
│   ├── 📄 Funds_stats_dict.json          # Fund configuration data
│   ├── 📄 Daily_trades_use.csv           # Daily trading data
│   └── utilities/                         # Core utilities
│       ├── 📄 Trading_routines_functions.py    # Main workflow manager
│       ├── 📄 Trading_allocation_calculation.py # Allocation calculations
│       ├── 📄 Solver_Allocation.py             # Optimization solver
│       ├── 📄 Auxiliar_function.py             # Helper functions
│       └── 📄 Post_trades_api.py               # API management
└── 📄 Daily_trades_use.csv               # Trading data file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud credentials configured
- Access to BigQuery datasets
- Trading API credentials

### Setup

1. **Clone or download the project**:
   ```bash
   cd Trading_Process
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Google Cloud credentials**:
   ```bash
   # Set up your Google Cloud credentials
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   ```

4. **Verify installation**:
   ```python
   from daily_trading_workflow import workflow_manager
   print("✅ Installation successful!")
   ```

## 🚀 Quick Start

### Basic Usage

```python
from daily_trading_workflow import workflow_manager

# Run complete workflow
workflow_manager.execute_daily_trading_workflow()

# Or run specific parts throughout the day:

# 🌅 Morning: Initial Orders
workflow_manager.run_initial_orders()

# 🏭 Mid-day: Primary Calculation  
workflow_manager.run_primary_calculation()

# 🔄 Afternoon: Adjustments
workflow_manager.run_adjustment_cycle()

# 🎯 Final: Complete Adjustment Workflow
workflow_manager.complete_adjustment_workflow()
```

### Simple Function Calls

```python
# Even simpler - use direct function calls
from daily_trading_workflow import (
    run_initial_orders,
    run_primary_calculation, 
    run_adjustment_cycle,
    complete_adjustment_workflow
)

# Morning routine
run_initial_orders()

# Afternoon routine  
run_adjustment_cycle()
```

## 📖 Usage

### Daily Workflow

The system is designed to be run in parts throughout the trading day:

#### 1. 🌅 Morning - Initial Orders
```python
run_initial_orders()
```
- Generates initial trading orders
- Creates email instructions for traders
- Organizes onshore/offshore trades

#### 2. 🏭 Mid-day - Primary Units
```python
run_primary_calculation()
```
- Calculates optimal primary unit allocations
- Saves data to BigQuery
- Generates primary unit requests

#### 3. 🔄 Afternoon - Adjustments
```python
run_adjustment_cycle(max_iterations=3)
```
- Runs iterative adjustment cycles
- Monitors execution vs targets
- Generates adjustment instructions

#### 4. 🎯 End of Day - Final Allocation
```python
complete_adjustment_workflow()
```
- Completes final adjustments
- Creates broker-ready orders
- Saves final allocations

### Advanced Usage

#### Custom Workflow Steps
```python
# Run specific workflow steps
results = workflow_manager.execute_daily_trading_workflow(
    step="primary",  # "initial", "primary", "adjustments", or "all"
    max_adjustment_iterations=5
)
```

#### Check Status
```python
# Quick status check without running cycles
status = workflow_manager.check_adjustment_status()
```

## 🔧 API Reference

### TradingWorkflowManager

Main class for managing the complete trading workflow.

#### Methods

- `run_initial_orders()` - Execute initial order generation
- `run_primary_calculation()` - Calculate primary unit allocations  
- `run_adjustment_cycle(max_iterations=3)` - Run adjustment cycles
- `complete_adjustment_workflow()` - Complete final adjustments
- `check_adjustment_status()` - Check current status

### TradeAllocationCalculator

Handles trade allocation calculations and optimizations.

### AllocationSolver

Advanced optimization solver for share allocations.

### TradingAPIManager

Manages API connections and data retrieval.

## ⚙️ Configuration

### Environment Variables

```bash
# Google Cloud Project
export GOOGLE_CLOUD_PROJECT="hdx-data-platform"

# BigQuery Dataset
export BIGQUERY_DATASET="team_trading"

# Trading API Secret
export TRADING_API_SECRET="hdx-routines-secret"
```

### Data Files

- `src/Funds_stats_dict.json` - Fund configuration and asset mappings
- `src/Daily_trades_use.csv` - Daily trading targets and strategies

### Supported Assets

**NASDAQ ETFs**: HASH11, SOLH11, BITH11, ETHE11, XRPH11  
**Thematic ETFs**: DEFI11, WEB311, META11, FOMO11

## 🤝 Contributing

### Development Setup

1. Follow installation steps above
2. Make changes to the codebase
3. Test thoroughly before deployment
4. Update documentation as needed

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Keep functions focused and modular

## 📞 Support

For questions or issues:
- Check the code documentation
- Review error logs in BigQuery
- Contact the trading team

## 📝 License

This project is proprietary and confidential. All rights reserved.

---

**⚡ Built for efficient, automated trading operations**

*Last updated: 2025*

## 🔒 Security Notice

This project uses environment variables to manage sensitive credentials. **NEVER commit the `.env` file or any files containing secrets to version control.**

## 🚀 Quick Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd Trading_Process
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root with your credentials:

```bash
# Copy the example and fill in your values
cp .env.example .env
```

Required environment variables:

```env
# Gmail Configuration
GMAIL_APP_PASSWORD=your_gmail_app_password
SENDER_EMAIL=your_email@company.com
RECIPIENT_EMAIL=recipient@company.com
CC_EMAILS=email1@company.com,email2@company.com

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=your-gcp-project
GOOGLE_CLOUD_PROJECT_ID_USERS=your-gcp-users-project

# AWS Configuration
API_SECRET_ARN=your-aws-secret-arn
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-aws-account-id
AWS_ROLE_NAME=your-role-name

# API Endpoints
INOA_BASE_URL=http://your-internal-api/
POSTTRADES_BASE_URL=https://your-trading-api.com
HASHDX_API_BASE_URL=https://your-api.com

# File Paths (adjust to your system)
CSV_FOLDER_PATH=/path/to/your/csv/folder
DAILY_TRADES_CSV_PATH=/path/to/daily_trades.csv
```

### 3. Verify Setup

```python
from dotenv import load_dotenv
import os

load_dotenv()
print("✅ Environment loaded successfully!")
```

## 📊 Usage

### Daily Workflow

```python
# 1. Morning - Initial Orders
result_initial = run_initial_orders()
send_initial_email(result_initial)

# 2. Mid-day - Primary Calculation
result_nasdaq = run_primary_calculation('nasdaq')

# 3. Afternoon - Adjustment Cycle
adjustments = run_adjustment_cycle(max_iterations=3)
send_adjustment_email(adjustments)

# 4. End of Day - Final Allocation
final_result = complete_adjustment_workflow()
```

## 🔧 Configuration

### Email Settings

- Uses Gmail SMTP with app passwords
- Supports HTML email templates
- Automatic CC to trading team

### Google Cloud Integration

- BigQuery for data storage
- Automatic table management
- User confirmation for uploads

### AWS Integration

- Secrets Manager for credential storage
- IAM role-based authentication

## 📁 Project Structure

```
Trading_Process/
├── src/
│   └── utilities/
│       ├── Email_sender.py          # Email management
│       ├── Trading_routines_functions.py  # Core workflows
│       ├── Auxiliar_function.py     # Helper functions
│       └── Post_trades_api.py       # API integrations
├── Daily_functions_new.ipynb        # Main workflow notebook
├── .env                            # Environment variables (DO NOT COMMIT)
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛡️ Security Best Practices

### Environment Variables
- All sensitive data is stored in `.env` file
- `.env` file is excluded from version control
- Default values provided for non-sensitive configs

### Git Security
- Comprehensive `.gitignore` excludes sensitive files
- No hardcoded credentials in source code
- CSV files and data exports are excluded

### Access Control
- AWS IAM roles for cloud access
- Google Cloud service accounts
- Email app passwords (not account passwords)

## 🚨 Before Committing to GitHub

1. ✅ Verify `.env` file is **NOT** tracked by git:
   ```bash
   git status
   # .env should NOT appear in the list
   ```

2. ✅ Check for any remaining secrets:
   ```bash
   grep -r "password\|secret\|key" src/ --exclude-dir=__pycache__
   # Should only show environment variable references
   ```

3. ✅ Verify `.gitignore` is working:
   ```bash
   git check-ignore .env
   # Should output: .env
   ```

## 📝 Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `GMAIL_APP_PASSWORD` | Gmail app-specific password | Yes |
| `SENDER_EMAIL` | Email address for sending | Yes |
| `RECIPIENT_EMAIL` | Primary recipient email | Yes |
| `CC_EMAILS` | Comma-separated CC emails | No |
| `GOOGLE_CLOUD_PROJECT_ID` | Main GCP project ID | Yes |
| `GOOGLE_CLOUD_PROJECT_ID_USERS` | Users GCP project ID | Yes |
| `API_SECRET_ARN` | AWS secrets ARN | Yes |
| `AWS_REGION` | AWS region | No (default: us-east-1) |
| `CSV_FOLDER_PATH` | Path for CSV exports | No (has default) |
| `DAILY_TRADES_CSV_PATH` | Daily trades CSV path | No (has default) |

## 🔍 Troubleshooting

### Common Issues

1. **Missing environment variables**:
   ```
   ValueError: Missing required environment variables
   ```
   Solution: Check your `.env` file has all required variables

2. **Gmail authentication error**:
   ```
   SMTPAuthenticationError: Username and Password not accepted
   ```
   Solution: Use Gmail app password, not account password

3. **Google Cloud authentication**:
   ```
   DefaultCredentialsError
   ```
   Solution: Run `gcloud auth application-default login`

## 📞 Support

For issues or questions, contact the Investment Management team.

---

**⚠️ Remember: Never commit sensitive credentials to version control!** 