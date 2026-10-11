### Funcion de Shwefel
- Se escogio esta funcion por su naturaleza extremista para hacer mas interesante el analisis y la comparacion entre ambas funciones.

### Descenso por gradiente
- Usamos la constante estandar del benchmar de schwefel

La funcion de schwefel no es tan buena para ser evaluada con el gradeinte, pero por que este paso es importante aunque falle?
sirva para mostrar que un metodo local no basta en funciones multimodales, deberia ser mejor en los bioinspirados

-El gradiente analitico de la funcion de schwefel suele ser problematico cerca de 0; el problema real es la derivada de $sqrt(abs(x_i)) en x_i = 0$ ya que no existe, es una cuspide, tiene pendiente infinita en el origen desde ambos lados.
- La funcion es continua y deribable en casi todo punto
- La formula analitica falla en xi=0 (division por 0)
- Cerca de 0 el gradiente puede ser inestable porque se estan calculando derivadas de una funcion con pico, no converge bien
- El optimo global esta en xi aprox 420.97, muy lejos de 0, pero puede pasar por zonas cercanas a 0 y tropezar


Para el descenso por gradiente es dificil salir de un minimo local pues solo mira la pendiente en el punto actual, es determinista, dada la condicion actual la trayectoria es unica, es greedy: siempre va cuesta abajo. No puede saltar a otro valle