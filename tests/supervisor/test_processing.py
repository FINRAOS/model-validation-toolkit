import copy

import pandas as pd
import pandas.testing

from mvtk.supervisor.processing import (
    replace_nulls,
    normalize_ts_columns,
)


def test_replace_nulls():
    for col_list in [["col1"], ["col2"], ["col1", "col2"]]:
        init_rows = [
            {"col1": "test1_1", "col2": "test1_2"},
            {"col1": None, "col2": "test2_2"},
            {"col1": "test3_1", "col2": None},
            {"col1": None, "col2": None},
        ]

        expect_rows = copy.deepcopy(init_rows)

        for i in range(0, len(expect_rows)):
            for col in col_list:
                if expect_rows[i][col] is None:
                    expect_rows[i][col] = "1"

        init_df = pd.DataFrame(init_rows)
        expect_df = pd.DataFrame(expect_rows)

        actual = replace_nulls(init_df, "1", col_list)
        expect = expect_df

        pandas.testing.assert_frame_equal(actual, expect)


def time_to_seconds(time):
    return int(time[:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:6])


def test_process_ts_columns():
    format_map = {"col2": "%H:%M:%S.%f", "col3": "%H%M%S.%f", "col4": "%H%M%S"}

    for col_list in [
        ["col2"],
        ["col3"],
        ["col4"],
        ["col2", "col3"],
        ["col2", "col4"],
        ["col3", "col4"],
        ["col2", "col3", "col4"],
    ]:
        init_rows = [
            {
                "col1": "test1",
                "col2": "10:11:12.123456",
                "col3": "101112.123456",
                "col4": "101112",
            },
            {
                "col1": "test2",
                "col2": None,
                "col3": "202123.123456",
                "col4": "202124",
            },
            {
                "col1": "test3",
                "col2": "10:31:32.123456",
                "col3": None,
                "col4": "103134",
            },
            {
                "col1": "test4",
                "col2": "20:41:42.123456",
                "col3": "204143.123456",
                "col4": None,
            },
        ]

        expect_rows = copy.deepcopy(init_rows)

        for i in range(0, len(expect_rows)):
            for col in col_list:
                if expect_rows[i][col] is None:
                    expect_rows[i][col] = -1
                else:
                    expect_rows[i][col] = str(
                        round(
                            time_to_seconds(expect_rows[i][col].replace(":", ""))
                            / 86400,
                            5,
                        )
                    )

        init_df = pd.DataFrame(init_rows)
        expect = pd.DataFrame(expect_rows)

        for col in ["col2", "col3", "col4"]:
            init_df[col] = pd.to_datetime(init_df[col], format=format_map[col])
            if col not in col_list:
                expect[col] = pd.to_datetime(expect[col], format=format_map[col])
            else:
                expect[col] = expect[col].astype(float)

        actual = normalize_ts_columns(init_df, col_list)

        pandas.testing.assert_frame_equal(actual, expect)
