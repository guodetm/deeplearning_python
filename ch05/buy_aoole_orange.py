from ch05.layer_naive import MulLayer, AddLayer

apple = 100
orange = 150
apple_num = 2
orange_num = 3
tax = 1.1

mul_app_layer = MulLayer()
mul_org_layer = MulLayer()
add_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_app_layer.forward(apple, apple_num)
orange_price = mul_org_layer.forward(orange, orange_num)
add_price = add_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(add_price, tax)
print(price)

# backward
dprice = 1
dadd_price, dtax = mul_tax_layer.backward(dprice)
dapple, dorange = add_layer.backward(dadd_price)
dorange_price, dorange_num = mul_org_layer.backward(dorange)
dapple_price, dapple_num = mul_app_layer.backward(dapple)
print(dtax, dapple_price, dapple_num, dorange_price, dorange_num)
