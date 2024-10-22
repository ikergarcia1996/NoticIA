def summarize_clickbait_short_prompt(
    headline: str,
    body: str,
) -> str:
    """
    Generate the prompt for the model.

    Args:
        headline (`str`):
            The headline of the article.
        body (`str`):
            The body of the article.
    Returns:
        `str`: The formatted prompt.
    """

    return (
        f"Ahora eres una Inteligencia Artificial experta en desmontar titulares sensacionalistas o clickbait. "
        f"Tu tarea consiste en analizar noticias con titulares sensacionalistas y "
        f"generar un resumen de una sola frase que revele la verdad detrás del titular.\n"
        f"Este es el titular de la noticia: {headline}\n"
        f"El titular plantea una pregunta o proporciona información incompleta. "
        f"Debes buscar en el cuerpo de la noticia una frase que responda lo que se sugiere en el título. "
        f"Siempre que puedas cita el texto original, especialmente si se trata de una frase que alguien ha dicho. "
        f"Si citas una frase que alguien ha dicho, usa comillas para indicar que es una cita. "
        f"Usa siempre las mínimas palabras posibles. No es necesario que la respuesta sea una oración completa, "
        f"puede ser sólo el foco de la pregunta. "
        f"Recuerda responder siempre en Español.\n"
        f"Este es el cuerpo de la noticia:\n"
        f"{body}\n"
    )


def summarize_clickbait_large_prompt(
    headline: str,
    body: str,
) -> str:
    """
    Generate the prompt for the model.

    Args:
        headline (`str`):
            The headline of the article.
        body (`str`):
            The body of the article.
    Returns:
        `str`: The formatted prompt.
    """

    return (
        f"Ahora eres una Inteligencia Artificial experta en desmontar titulares sensacionalistas o clickbait. "
        f"Tu tarea consiste en analizar noticias con titulares sensacionalistas y "
        f"generar un resumen de una sola frase que revele la verdad detrás del titular.\n"
        f"Este es el titular de la noticia: {headline}\n"
        f"El titular plantea una pregunta o proporciona información incompleta. "
        f"Debes buscar en el cuerpo de la noticia una frase que responda lo que se sugiere en el título. "
        f"Siempre que puedas cita el texto original, especialmente si se trata de una frase que alguien ha dicho. "
        f"Recuerda responder siempre en Español.\n"
        f"Este es el cuerpo de la noticia:\n"
        f"{body}\n"
    )


def summarize_prompt(
    headline: str,
    body: str,
) -> str:
    """
    Generate the prompt for the model.

    Args:
        headline (`str`):
            The headline of the article.
        body (`str`):
            The body of the article.
    Returns:
        `str`: The formatted prompt.
    """

    return (
        f"Ahora eres una Inteligencia Artificial experta en resumir noticias. "
        f"Este es el titular de la noticia: {headline}\n"
        f"Por favor, genera un resumen corto de la noticia. Recuerda responder siempre en Español.\n"
        f"Este es el cuerpo de la noticia:\n"
        f"{body}\n"
    )


def clickbait_prompt_flor(
    headline: str,
    body: str,
) -> str:
    """
    Specific prompt for FLOR-6.3B-Instructed which uses a prompt format that is difficult to adapt,
    into a jinja template.

    Args:
        headline (`str`):
            The headline of the article.
        body (`str`):
            The body of the article.
    Returns:
        `str`: The formatted prompt.
    """

    return (
        f"### Instruction\n"
        f"Ahora eres una Inteligencia Artificial experta en desmontar titulares sensacionalistas o clickbait. "
        f"Tu tarea consiste en analizar noticias con titulares sensacionalistas y "
        f"generar un resumen de una sola frase que revele la verdad detrás del titular.\n"
        f"Este es el titular de la noticia: {headline}\n"
        f"El titular plantea una pregunta o proporciona información incompleta. "
        f"Debes buscar en el cuerpo de la noticia una frase que responda lo que se sugiere en el título. "
        f"Siempre que puedas cita el texto original, especialmente si se trata de una frase que alguien ha dicho. "
        f"Si citas una frase que alguien ha dicho, usa comillas para indicar que es una cita. "
        f"Usa siempre las mínimas palabras posibles. No es necesario que la respuesta sea una oración completa. "
        f"Puede ser sólo el foco de la pregunta. "
        f"Recuerda responder siempre en Español.\n"
        f"Este es el cuerpo de la noticia:\n"
        f"### Context\n"
        f"{body}\n"
        f"### Answer\n"
    )
