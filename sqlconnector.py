import configparser
import pg8000

config = configparser.ConfigParser()
config.read("config.ini")

DB_HOST = config["postgresql"]["host"]
DB_PORT = int(config["postgresql"]["port"])
DB_NAME = config["postgresql"]["database"]
DB_USER = config["postgresql"]["username"]
DB_PASS = config["postgresql"]["password"]

def getconn() -> pg8000.Connection:
    """
    Helper function to return SQL connection.
    """
    return pg8000.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
