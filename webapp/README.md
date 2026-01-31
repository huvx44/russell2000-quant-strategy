# Russell 2000 Quant Strategy - Web Application

Web interface for the Russell 2000 quantitative trading strategy, designed to run on a Synology NAS or any Docker host.

## Features

- **Backtest** - Configure and run backtests with real-time output streaming via WebSocket
- **Results** - Browse results folders, view charts, inspect CSV data with sortable tables, export to Excel
- **Configuration** - Adjust factor weights and constraints, save/load presets
- **Portfolio** - Track real holdings, refresh prices, P&L charts, tag/compare portfolio versions
- **Authentication** - Password login + WebAuthn passkey support (fingerprint/Face ID)

## Quick Start

### Docker (recommended for Synology NAS)

```bash
cd webapp

# Copy and edit environment file
cp .env.example .env
# Edit .env with your settings (SECRET_KEY, ADMIN_PASSWORD, etc.)

# Build and start
docker compose up -d

# Access at http://localhost:5000
# Default login: admin / admin
```

### Local Development

```bash
cd webapp
pip install -r requirements.txt
python wsgi.py
# Access at http://localhost:5000
```

## Synology NAS Setup

### 1. Install Container Manager

Open Package Center on your Synology and install "Container Manager" (Docker).

### 2. Upload project files

Copy the entire `russell2000-quant-strategy` folder to your NAS shared folder (e.g., `/volume1/docker/quant-strategy/`).

### 3. Configure environment

```bash
cd /volume1/docker/quant-strategy/webapp
cp .env.example .env
```

Edit `.env`:
```
SECRET_KEY=<generate-a-random-string>
ADMIN_PASSWORD=<your-secure-password>
RP_ID=nas.yourdomain.com
RP_ORIGIN=https://nas.yourdomain.com
```

### 4. Build and start

Using Container Manager UI or SSH:
```bash
cd /volume1/docker/quant-strategy/webapp
docker compose up -d
```

### 5. Set up HTTPS reverse proxy (required for passkeys)

1. Open Synology **Control Panel** > **Login Portal** > **Advanced** > **Reverse Proxy**
2. Create a new rule:
   - Source: HTTPS, your hostname, port 443
   - Destination: HTTP, localhost, port 5000
3. Enable HTTPS certificate (Let's Encrypt) in **Security** > **Certificate**

WebAuthn passkeys require HTTPS. Without the reverse proxy, only password login will work.

## Architecture

```
webapp/
├── wsgi.py                    # WSGI entry point
├── app/
│   ├── __init__.py            # Flask app factory
│   ├── auth/                  # Authentication (password + WebAuthn)
│   │   ├── models.py          # User & WebAuthnCredential models
│   │   └── routes.py          # Login, logout, passkey registration
│   ├── backtest/              # Backtest execution
│   │   ├── routes.py          # Backtest page
│   │   ├── events.py          # SocketIO event handlers
│   │   └── runner.py          # Subprocess management
│   ├── results/               # Results viewer
│   │   └── routes.py          # Folder browser, CSV viewer, chart serving
│   ├── config_mgr/            # Configuration management
│   │   └── routes.py          # Preset CRUD
│   ├── portfolio/             # Portfolio manager
│   │   └── routes.py          # Holdings CRUD, charts, tags
│   ├── templates/             # Jinja2 HTML templates
│   └── static/css/            # Stylesheets
├── data/                      # Persistent data (Docker volume)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `change-me...` | Flask session encryption key |
| `ADMIN_PASSWORD` | `admin` | Initial admin password (first launch only) |
| `RP_ID` | `localhost` | WebAuthn Relying Party ID (your hostname) |
| `RP_NAME` | `R2K Quant Strategy` | WebAuthn display name |
| `RP_ORIGIN` | `http://localhost:5000` | Full origin URL for WebAuthn |
| `DATA_DIR` | `./data` | Path for persistent data |
| `CORE_DIR` | `../` | Path to backtest scripts |

### Data Persistence

All mutable data is stored in the `data/` directory:
- `app.db` - SQLite database (user accounts, passkey credentials)
- `my_portfolio.csv` - Current portfolio holdings
- `portfolio_history.csv` - Portfolio snapshots
- `portfolio_tags.csv` - Tag index
- `tagged_portfolios/` - Saved portfolio versions
- `config_presets/` - Strategy configuration presets

Cache and results are stored in the core directory alongside the backtest scripts.
