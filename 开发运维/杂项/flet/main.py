import flet as ft
from views.table import getTableView
from views.rail import getRailView
def main(page: ft.Page):
    page.title = "Routes Example"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    
    rail=getRailView(page)
    

    def route_change(route):
        page.views.clear()
        troute=ft.TemplateRoute(route.route)
        if troute.match("/account/:account_id/orders/:order_id"):
            print("Account:", troute.account_id, "Order:", troute.order_id)
            page.views.append(
                ft.View(
                    route.route,
                    [
                        ft.AppBar(title=ft.Text("Account"), bgcolor=ft.colors.SURFACE_VARIANT),
                        ft.ElevatedButton("Go Home", on_click=lambda _: page.go("/"))
                    ]
                )
            )

        if route.route == "/":
            
            page.views.append(
                ft.View(
                    route.route,
                    [
                        ft.AppBar(title=ft.Text("Flet app"), bgcolor=ft.colors.SURFACE_VARIANT),
                        ft.Row(
                            [
                                rail,
                                ft.VerticalDivider(width=1),
                                ft.Column([ ft.Text("Body!"),
                                    ft.ElevatedButton("Visit Store", on_click=lambda _: page.go("/store")),
                                    ft.ElevatedButton("Account Orders", on_click=lambda _: page.go("/account/me/orders/1")),
                                    ], alignment=ft.MainAxisAlignment.START, expand=True),
                            ],
                            expand=True,
                        ),
                        
                    ],
                )
            )
        if route.route == "/store":
            page.views.append(
                ft.View(
                    route.route,
                    [
                        ft.AppBar(title=ft.Text("Store"), bgcolor=ft.colors.SURFACE_VARIANT),
                        *getTableView(page)
                    ],
                )
            )
        page.update()

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.go(page.route)
    


ft.app(target=main)