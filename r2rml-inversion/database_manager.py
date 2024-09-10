import time
import docker
from docker.errors import NotFound
import subprocess
import psutil


class DatabaseManager:
    def __init__(self):
        self.client = docker.from_env()
        self.containers = {}
        self.ports = {'postgresql': 5432, 'mysql': 3306}

    def get_container(self, database_system):
        if database_system not in self.containers or not self.container_is_running(database_system):
            self.start_container(database_system)
        return self.containers[database_system]

    def start_container(self, database_system):
        self.stop_existing_services(database_system)
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                image = 'postgres:13' if database_system == 'postgresql' else 'mysql:8'
                port = self.ports[database_system]
                container = self.client.containers.run(
                    image,
                    detach=True,
                    remove=True,
                    environment={
                        'POSTGRES_PASSWORD': 'r2rml',
                        'POSTGRES_USER': 'r2rml',
                        'POSTGRES_DB': 'r2rml'
                    } if database_system == 'postgresql' else {
                        'MYSQL_ROOT_PASSWORD': 'r2rml',
                        'MYSQL_USER': 'r2rml',
                        'MYSQL_PASSWORD': 'r2rml',
                        'MYSQL_DATABASE': 'r2rml'
                    },
                    ports={f'{port}/tcp': port}
                )
                print(f"Container started for {database_system}: {container.id}")
                self.containers[database_system] = container
                time.sleep(10)  # Wait for the database to be ready
                return
            except docker.errors.APIError as e:
                if "port is already allocated" in str(e) and attempt < max_attempts - 1:
                    print(f"Port {port} is still in use, attempting to stop services again...")
                    self.stop_existing_services(database_system)
                    continue
                else:
                    print(f"Error starting container: {e}")
                    raise
        raise Exception(f"Failed to start {database_system} container after {max_attempts} attempts")

    def stop_existing_services(self, database_system):
        port = self.ports[database_system]
        
        # Stop Docker containers using the port
        containers = self.client.containers.list()
        for container in containers:
            container_ports = container.attrs['NetworkSettings']['Ports']
            for container_port, host_ports in container_ports.items():
                if host_ports and int(host_ports[0]['HostPort']) == port:
                    print(f"Stopping Docker container {container.id} using port {port}")
                    container.stop()
                    container.remove()

        # Stop system service (if running)
        if database_system == 'postgresql':
            self._stop_postgresql_service()
        elif database_system == 'mysql':
            self._stop_mysql_service()

        # Kill any remaining processes using the port
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                try:
                    process = psutil.Process(conn.pid)
                    print(f"Killing process {process.pid} using port {port}")
                    process.terminate()
                    process.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass

    def _stop_postgresql_service(self):
        try:
            subprocess.run(['sudo', 'service', 'postgresql', 'stop'], check=True)
            print("Stopped PostgreSQL system service")
        except subprocess.CalledProcessError:
            print("Failed to stop PostgreSQL system service")

    def _stop_mysql_service(self):
        try:
            subprocess.run(['sudo', 'service', 'mysql', 'stop'], check=True)
            print("Stopped MySQL system service")
        except subprocess.CalledProcessError:
            print("Failed to stop MySQL system service")

    def container_is_running(self, database_system):
        if database_system in self.containers:
            try:
                self.containers[database_system].reload()
                return self.containers[database_system].status == 'running'
            except NotFound:
                return False
        return False

    def reset_database(self, database_system):
        container = self.get_container(database_system)
        if database_system == 'postgresql':
            container.exec_run("psql -U r2rml -c 'DROP SCHEMA public CASCADE; CREATE SCHEMA public;'")
        else:  # MySQL
            container.exec_run("mysql -u r2rml -pr2rml -e 'DROP DATABASE r2rml; CREATE DATABASE r2rml;'")

    def get_connection_string(self, database_system):
        port = self.ports[database_system]
        if database_system == 'postgresql':
            return f"postgresql://r2rml:r2rml@localhost:{port}/r2rml"
        else:
            return f"mysql://r2rml:r2rml@localhost:{port}/r2rml"