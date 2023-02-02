import ares.static.DarkMatterHeating as DarkMatterHeating

z = 404.5000000005507
Tk = 1089.184155018262
Tchi = 4.9441439895944095
xe = 0.0009684253634894803
v_chi_b = 5260.93867634403
interaction = DarkMatterHeating.baryon_dark_matter_interaction(z, Tk, Tchi, xe, v_chi_b)
print(interaction)