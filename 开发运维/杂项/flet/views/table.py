import flet as ft

table=ft.DataTable(
    columns=[
        ft.DataColumn(ft.Text("名字")),
        ft.DataColumn(ft.Text("姓氏")),
        ft.DataColumn(ft.Text("年龄"), numeric=True),
    ],
    rows=[
        ft.DataRow(
            cells=[
                ft.DataCell(ft.Text("约翰")),
                ft.DataCell(ft.Text("史密斯")),
                ft.DataCell(ft.Text("43")),
            ],
        ),
        ft.DataRow(
            cells=[
                ft.DataCell(ft.Text("杰克")),
                ft.DataCell(ft.Text("布朗")),
                ft.DataCell(ft.Text("19")),
            ],
        ),
        ft.DataRow(
            cells=[
                ft.DataCell(ft.Text("爱丽丝")),
                ft.DataCell(ft.Text("王")),
                ft.DataCell(ft.Text("25")),
            ],
        ),
    ],
)

def getTableView(page):
    view=[
        table,
        ft.ElevatedButton("Go Home", on_click=lambda _: page.go("/")),
    ]
    return view