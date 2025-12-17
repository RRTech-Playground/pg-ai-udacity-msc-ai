from shirt import Shirt

shirt_one = Shirt('red', 'M', 'long_sleeved', 45)
shirt_two = Shirt('orange', 'S', 'short_sleeved', 30)

print(shirt_one.price)
print(shirt_one.color)

shirt_two.change_price(45)
print(shirt_two.price)
