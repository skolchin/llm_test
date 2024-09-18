from functools import cached_property
from langchain_community.utilities.sql_database import SQLDatabase
from typing import Iterable, Mapping, Optional, List

class SQLDatabaseExt(SQLDatabase):

    @cached_property
    def table_comments(self) -> Mapping[str, str]:
        return {self._metadata.tables[t].name: f'{self._metadata.tables[t].comment}' for t in self._metadata.tables}

    @cached_property
    def comments_tables(self) -> Mapping[str, str]:
        return {f'{self._metadata.tables[t].comment}': self._metadata.tables[t].name for t in self._metadata.tables}

    def get_usable_table_names(self) -> Iterable[str]:
        tables = super().get_usable_table_names()
        if not hasattr(self, '_metadata'):
            return tables
        return list(tables) + list(self.table_comments.values())

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        real_table_names = None
        if table_names:
            real_table_names = [self.comments_tables.get(t, t) for t in table_names]
        return super().get_table_info(real_table_names)

    def run(self, *args, **kwargs):
        command = kwargs.pop('command', None)
        if not command:
            command = args[0]
            args = args[1:]
        for comment, table in self.comments_tables.items():
            command = command.replace(comment, table)
        return super().run(command, *args, **kwargs)

# import os
# from dotenv import load_dotenv
# from sqlalchemy import create_engine

# engine = create_engine(
#     'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/actuals'.format(
#         POSTGRES_USER=os.environ['POSTGRES_USER'],
#         POSTGRES_PASSWORD=os.environ['POSTGRES_PASSWORD'],
#     )
# )
#
# load_dotenv()
#
# db = SQLDatabaseExt(engine, 'eis')
# print(db.get_usable_table_names())
# print(db.get_table_info(['"таблица Документы ОПС_21"']))
# print(db.run('select * from "таблица Документы ОПС_21"'))
