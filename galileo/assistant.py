import pandas as pd
import numpy as np
import logging

from typing import List, Dict
from IPython.display import display


class Assistant:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    def _calculate_duplicates(self, keys) -> int:
        return len(self.dataframe[self.dataframe.duplicated(keys) == True])

    @staticmethod
    def _get_conversion_dict() -> Dict[str, type]:

        conversor = {
            "Int32": float,
            "int32": float,
            "int64": float,
            "Int64": float,
            "float64": float,
            "float32": float,
            "string": str,
            "category": str,
            "object": str,
        }

        return conversor

    @staticmethod
    def _convert_column_types(dataframe) -> pd.DataFrame:
        """Padroniza os tipos das colunas."""
        instructions = Assistant._get_conversion_dict()

        conv = {
            col: instructions[type_.name]
            for col, type_ in dataframe.dtypes.to_dict().items()
        }
        return dataframe.astype(conv)

    def check_duplicates(
        self, keys: List[str], remove_duplicates: bool = False
    ) -> pd.DataFrame:
        """Checa se há duplicações no dataframe."""

        try:
            duplication_size = self._calculate_duplicates(keys=keys)
            #  Taking the first column of keys as the identification column
            id = keys[0]

            if duplication_size > 0:
                example_id = self.dataframe[self.dataframe.duplicated(keys)][id].iloc[1]
                evidence = self.dataframe[self.dataframe[id] == example_id]
                print(f"Total duplications: {duplication_size}\n")
                print("Evidence:\n")
                display(evidence)

                if remove_duplicates:
                    print("Removing duplicates...")
                    self.dataframe = self.dataframe.drop_duplicates(subset=keys)
                    self.dataframe = Assistant._convert_column_types(self.dataframe)
                    duplication_size = self._calculate_duplicates(keys=keys)
                    print(f"Total duplications after processing: {duplication_size}")
                    return self.dataframe
                else:
                    print(
                        "Found duplications but remove_duplicates argument is set to False"
                    )
                    return Assistant._convert_column_types(self.dataframe)
            else:
                print("No duplications found for given keys")
                return self.dataframe
        except Exception as e:
            print("Processed failed due to the following error:\n")
            print(f"{str(e)}\n")
            print("Rolling back to the original dataframe")
            return Assistant._convert_column_types(self.dataframe)
