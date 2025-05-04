### Descripción del proyecto
La idea principal es construir una herramienta que, a partir del archivo de código de una clase Java y su contexto de proyecto, genere automáticamente un conjunto de tests unitarios relevantes. Para lograrlo, utilizamos un enfoque RAG donde, dada una clase de entrada, se recuperan clases similares desde una base vectorial construida a partir de proyectos reales obtenidos de GitHub.

### Input de la aplicación
El archivo de código de la clase Java objetivo.
El path del proyecto completo (para tener contexto adicional si es necesario durante la generación).
### Output de la aplicación
Un archivo o bloque de código que contiene los tests unitarios generados para la clase de entrada.
### ¿Qué van a persistir en la base de datos vectorial?
Vamos a almacenar embeddings de descripciones textuales de clases Java. Cada clase estará asociada a:

Una breve descripción funcional generada automáticamente.
El contenido de la clase (como texto).
Un conjunto de tests unitarios existentes vinculados a esa clase.
La descripción se usa para facilitar la búsqueda semántica de clases similares en la base vectorial. Esta descripción va a ser generada por una LLM.

### ¿De dónde obtienen los datos? (web scraping, documentos, generación manual, etc) ¿Cuál es el o los formatos?
Los datos provienen de proyectos open source extraídos desde GitHub. Los repos fueron buscados a mano o obtenidos de proyectos de la universidad en donde utilizamos java conociendo que teniamos buen test coverage. Los archivos fuente y los tests se almacenan junto con las descripciones en un JSON de formato.

``{
    clase:"",
    tests:"",
    descripcion:""
}``

### ¿Dónde ven el principal desafío?
El mayor desafío está en lograr que el modelo no solo genere tests sintácticamente correctos, sino también funcionales y coherentes con el comportamiento esperado de la clase. Esto implica combinar bien el retrieval con la generación, además de lidiar con dependencias, mocks, y estilos de testing variables entre proyectos. También va a ser clave manejar bien la representación semántica de las clases para que las búsquedas realmente devuelvan ejemplos útiles.


Adjunto algunos de los repos que vimos para usar como data sets



https://github.com/DanAg278/Java-Unit-Testing

https://github.com/christian-kesler/junit-testing-java

https://github.com/rieckpil/java-testing-toolbox
