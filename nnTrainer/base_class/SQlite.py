import sqlite3

class SQlite3Executor():
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

        self.connection = self.get_connection()

        self.cursor = self.get_cursor()

        self.execute_simple('PRAGMA foreign_keys = ON;')

    def get_connection(self) -> sqlite3.Connection:
        '''
        Stablish database connection
        '''
        return sqlite3.connect(self.database_path)
    
    def get_cursor(self) -> sqlite3.Cursor:
        '''
        Create database cursor
        '''
        return self.connection.cursor()

    def status(self) -> None:
        '''
        Show how many table rows has been changed since connection
        '''
        print(self.connection.total_changes)

    def execute_simple(self, query: str) -> None:
        '''
        Execute SQL queries
        '''
        self.cursor.execute(query)

    def execute_parameters(self, query: str, values: tuple) -> None:
        '''
        Execute SQL queries
        '''
        self.cursor.execute(query, values)

    def retrieve(self) -> list:
        return self.cursor.fetchall()

    def commit(self) -> None:
        '''
        Save changes to database
        '''
        self.connection.commit()

    def close(self) -> None:
        '''
        Close connection
        '''
        self.connection.close()