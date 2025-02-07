package main

import (
	"syscall/js"
)

// 导出add函数
func add(this js.Value, inputs []js.Value) interface{} {
	a := inputs[0].Int()
	b := inputs[1].Int()
	return a + b
}

// 导出add函数
func mul(this js.Value, inputs []js.Value) interface{} {
	a := inputs[0].Int()
	b := inputs[1].Int()
	return a * b
}

func main() {
	c := make(chan struct{}, 0)

	// 注册add函数
	js.Global().Set("add", js.FuncOf(add))
	js.Global().Set("mul", js.FuncOf(mul))

	<-c
}
