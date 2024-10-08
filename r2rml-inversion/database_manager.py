import time
import docker
from docker.errors import NotFound, APIError
import subprocess
import psutil
import pandas as pd
from sqlalchemy import create_engine


class DatabaseManager:
    def __init__(self):
        self.client = docker.from_env()
        self.containers = {}
        self.ports = {'postgresql': 5432, 'mysql': 3306, 'graphdb': 7200, 'dest_postgresql': 5433}
        self.graphdb_initialized = False

    def create_engine(self, connection_string):
        return create_engine(connection_string)

    def get_container(self, database_system):
        if database_system not in self.containers or not self.container_is_running(database_system):
            self.start_container(database_system)
        return self.containers[database_system]

    def start_container(self, database_system):
        self.stop_existing_services(database_system)
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if database_system == 'graphdb':
                    return self.start_graphdb_container()
                
                image = 'postgres:13' if database_system in ['postgresql', 'dest_postgresql'] else 'mysql:8'
                port = self.ports[database_system]
                
                environment = {
                    'POSTGRES_PASSWORD': 'r2rml',
                    'POSTGRES_USER': 'r2rml',
                    'POSTGRES_DB': 'r2rml'
                } if database_system in ['postgresql', 'dest_postgresql'] else {
                    'MYSQL_ROOT_PASSWORD': 'r2rml',
                    'MYSQL_USER': 'r2rml',
                    'MYSQL_PASSWORD': 'r2rml',
                    'MYSQL_DATABASE': 'r2rml'
                }
                
                if database_system == 'dest_postgresql':
                    # Use a custom command to change the port
                    command = f"-c port={port}"
                    container = self.client.containers.run(
                        image,
                        command=command,
                        detach=True,
                        remove=True,
                        environment=environment,
                        ports={f'{port}/tcp': port}
                    )
                else:
                    container = self.client.containers.run(
                        image,
                        detach=True,
                        remove=True,
                        environment=environment,
                        ports={f'{port}/tcp': port}
                    )
                
                print(f"Container started for {database_system}: {container.id}")
                self.containers[database_system] = container
                time.sleep(5)
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

    def start_graphdb_container(self):
        if not self.graphdb_initialized:
            port = self.ports['graphdb']
            container = self.client.containers.run(
                'ontotext/graphdb:10.7.3',
                detach=True,
                remove=True,
                ports={f'{port}/tcp': port},
                environment={
                    'GDB_JAVA_OPTS': '-Xmx2g -Xms2g',
                    'GDB_HEAP_SIZE': '2g'
                }
            )
            print(f"GraphDB container started: {container.id}")
            self.containers['graphdb'] = container
            self.graphdb_initialized = True
        return self.containers['graphdb']

    def stop_existing_services(self, database_system):
        port = self.ports[database_system]
        
        # Stop Docker containers using the port
        containers = self.client.containers.list()
        for container in containers:
            container_ports = container.attrs['NetworkSettings']['Ports']
            for container_port, host_ports in container_ports.items():
                if host_ports and int(host_ports[0]['HostPort']) == port:
                    print(f"Stopping Docker container {container.id} using port {port}")
                    try:
                        container.stop(timeout=10)
                        container.remove(force=True)
                    except APIError as e:
                        if 'removal of container' in str(e) and 'is already in progress' in str(e):
                            print(f"Container {container.id} is already being removed. Continuing...")
                        else:
                            raise

        # Stop system service (if running)
        if database_system == 'postgresql':
            self._stop_postgresql_service()
        elif database_system == 'mysql':
            self._stop_mysql_service()

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
        if database_system in ['postgresql', 'dest_postgresql']:
            # Terminate all connections
            container.exec_run("""
                psql -U postgres -c "
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = 'r2rml'
                AND pid <> pg_backend_pid();
                "
            """)
            # Drop and recreate the database
            container.exec_run("""
                psql -U postgres -c "
                DROP DATABASE IF EXISTS r2rml;
                CREATE DATABASE r2rml;
                "
            """)
            # Reconnect and set permissions
            container.exec_run("""
                psql -U postgres -d r2rml -c "
                CREATE SCHEMA IF NOT EXISTS public;
                GRANT ALL ON SCHEMA public TO r2rml;
                GRANT ALL ON SCHEMA public TO public;
                ALTER DATABASE r2rml OWNER TO r2rml;
                "
            """)
        elif database_system == 'mysql':
            container.exec_run("""
                mysql -u root -pr2rml -e '
                DROP DATABASE IF EXISTS r2rml;
                CREATE DATABASE r2rml;
                GRANT ALL PRIVILEGES ON r2rml.* TO 'r2rml'@'%';
                FLUSH PRIVILEGES;
                '
            """)
        elif database_system == 'graphdb':
            pass

        print(f"Database {database_system} has been reset.")

    def get_connection_string(self, database_system):
        port = self.ports[database_system]
        if database_system in ['postgresql', 'dest_postgresql']:
            return f"postgresql://r2rml:r2rml@localhost:{port}/r2rml"
        elif database_system == 'mysql':
            return f"mysql+pymysql://r2rml:r2rml@localhost:{port}/r2rml"
        elif database_system == 'graphdb':
            return f"http://localhost:{port}"

    def get_database_content(self, database_system):
        connection_string = self.get_connection_string(database_system)
        engine = self.create_engine(connection_string)
        
        try:
            with engine.connect() as connection:
                if database_system in ['postgresql', 'dest_postgresql']:
                    table_query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';"
                else:  # MySQL
                    table_query = "SHOW TABLES;"
                
                tables = pd.read_sql(table_query, connection)
                table_names = tables.values.flatten()
                
                db_content = {}
                for table in table_names:
                    db_content[table] = self.get_table_content(database_system, table)
                
                return db_content
        except Exception as e:
            print(f"Error getting database content: {str(e)}")
            return None
        finally:
            engine.dispose()

    def get_table_content(self, database_system, table_name):
        connection_string = self.get_connection_string(database_system)
        engine = self.create_engine(connection_string)
        
        try:
            with engine.connect() as connection:
                if database_system in ['postgresql', 'dest_postgresql']:
                    content_query = f'SELECT * FROM "{table_name}";'
                    datatype_query = f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}';
                    """
                else:  # MySQL
                    content_query = f"SELECT * FROM `{table_name}`;"
                    datatype_query = f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' AND table_schema = DATABASE();
                    """
                
                content = pd.read_sql(content_query, connection)
                datatypes = pd.read_sql(datatype_query, connection)
                
                if datatypes.empty:
                    return None
                
                datatypes = datatypes.set_index('column_name')['data_type']
                
                return {
                    'columns': content.columns.tolist(),
                    'data': content.values.tolist()
                }
        except Exception as e:
            print(f"Error getting table content: {str(e)}")
            return None
        finally:
            engine.dispose()