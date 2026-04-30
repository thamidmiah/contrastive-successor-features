"""A `dowel.logger.LogOutput` for CSV files."""
import csv
import io
import os
import warnings

from dowel import TabularInput
from dowel.simple_outputs import FileOutput
from dowel.utils import colorize


class CsvOutput(FileOutput):
    """CSV file output for logger.

    Dynamically expands columns when new keys appear (e.g. when training
    metrics start being logged after the replay buffer fills up).

    :param file_name: The file this output should log to.
    """

    def __init__(self, file_name):
        super().__init__(file_name)
        self._writer = None
        self._fieldnames = None
        self._warned_once = set()
        self._disable_warnings = False
        self._file_name = file_name
        self._rows = []  # keep all rows so we can rewrite if headers change

    @property
    def types_accepted(self):
        """Accept TabularInput objects only."""
        return (TabularInput, )

    def record(self, data, prefix=''):
        """Log tabular data to CSV."""
        if isinstance(data, TabularInput):
            to_csv = data.as_primitive_dict

            if not to_csv.keys() and not self._writer:
                return

            if not self._writer:
                # First write — set up headers
                self._fieldnames = set(to_csv.keys())
                self._writer = csv.DictWriter(
                    self._log_file,
                    fieldnames=sorted(list(self._fieldnames)),
                    extrasaction='ignore')
                self._writer.writeheader()

            new_keys = set(to_csv.keys()) - self._fieldnames
            if new_keys:
                # New columns appeared — rewrite entire CSV with expanded headers
                self._fieldnames = self._fieldnames | new_keys
                self._rewrite_csv_with_new_headers()

            self._rows.append(dict(to_csv))
            self._writer.writerow(to_csv)
            self._log_file.flush()

            for k in to_csv.keys():
                data.mark(k)
        else:
            raise ValueError('Unacceptable type.')

    def _rewrite_csv_with_new_headers(self):
        """Rewrite the CSV file with the expanded set of fieldnames."""
        sorted_fields = sorted(list(self._fieldnames))

        # Rewrite to a temporary string buffer, then overwrite the file
        self._log_file.close()

        with open(self._file_name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fields,
                                    extrasaction='ignore')
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

        # Re-open in append mode for future writes
        self._log_file = open(self._file_name, 'a', newline='')
        self._writer = csv.DictWriter(
            self._log_file,
            fieldnames=sorted_fields,
            extrasaction='ignore')

    def _warn(self, msg):
        """Warns the user using warnings.warn.

        The stacklevel parameter needs to be 3 to ensure the call to logger.log
        is the one printed.
        """
        if not self._disable_warnings and msg not in self._warned_once:
            warnings.warn(
                colorize(msg, 'yellow'), CsvOutputWarning, stacklevel=3)
        self._warned_once.add(msg)
        return msg

    def disable_warnings(self):
        """Disable logger warnings for testing."""
        self._disable_warnings = True


class CsvOutputWarning(UserWarning):
    """Warning class for CsvOutput."""

    pass
