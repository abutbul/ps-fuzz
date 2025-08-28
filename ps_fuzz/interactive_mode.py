from .app_config import AppConfig
from .langchain_integration import get_langchain_chat_models_info
from .prompt_injection_fuzzer import run_fuzzer
from .prompt_injection_fuzzer import run_interactive_chat
import inquirer
import colorama
# Use prompt_toolkit's ability to present a working multi-line editor
from prompt_toolkit import prompt as prompt_toolkit_prompt, HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
import logging
logger = logging.getLogger(__name__)

def multi_line_editor(initial_text: str) -> str:
    bindings = KeyBindings()

    @bindings.add('c-e')
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)

    print(f"{colorama.Style.BRIGHT}Edit prompt:{colorama.Style.RESET_ALL}")

    # Prompt for input using the session
    try:
        return prompt_toolkit_prompt(
            HTML("<prompt>This is a multi-line text editor. Press</prompt> <hl>Ctrl+E</hl> <prompt>to finish editing and save the prompt.</prompt> <hl>Ctrl+C</hl> <prompt>to cancel and leave the original prompt intact.</prompt>\n"),
            default=initial_text,
            multiline=True,
            key_bindings=bindings,
            style=Style.from_dict({
                '':        'fg:ansicyan',
                'prompt':  'fg:orange',
            }),
        )
    except KeyboardInterrupt:
        print(f"{colorama.Fore.RED}Edit cancelled by user. Leaving old system prompt intact.{colorama.Style.RESET_ALL}")
        return initial_text

def show_all_config(state: AppConfig):
    state.print_as_table()

class MainMenu:
    # Used to recall the last selected item in this menu between invocations (for convenience)
    selected = None
    
    @classmethod
    def show(cls, state: AppConfig):
        title = "Main Menu: What would you like to do today?"
        options = [
            ['Start Fuzzing your system prompt', run_fuzzer, MainMenu],
            ['Try your system prompt in the playground', run_interactive_chat, MainMenu],
            ['Edit System Prompt', None, SystemPromptEditor],
            ['Fuzzer Configuration', None, FuzzerOptions],
            ['Target LLM Configuration', None, TargetLLMOptions],
            ['Attack LLM Configuration', None, AttackLLMOptions],
            ['Embedding Configuration', None, EmbeddingOptions],
            ['Show all configuration', show_all_config, MainMenu],
            ['Exit', None, None],
        ]
        result = inquirer.prompt([
            inquirer.List(
                'action',
                message=title,
                choices=[x[0] for x in options],
                default=cls.selected
            )
        ])
        if result is None: return  # Handle prompt cancellation concisely
        func = {option[0]: option[1] for option in options}.get(result['action'], None)
        if func: func(state)
        cls.selected = result['action']
        return {option[0]: option[2] for option in options}.get(cls.selected, None)

class SystemPromptEditor:
    @classmethod
    def show(cls, state: AppConfig):
        print("System Prompt Editor: Edit the system prompt used by the fuzzer")
        print("---------------------------------------------------------------")
        state.system_prompt = multi_line_editor(state.system_prompt)
        return MainMenu

class FuzzerOptions:
    @classmethod
    def show(cls, state: AppConfig):
        print("Fuzzer Options: Review and modify the fuzzer options")
        print("----------------------------------------------------")
        result = inquirer.prompt([
            inquirer.Text('num_attempts',
                message="Number of fuzzing prompts to generate for each attack",
                default=str(state.num_attempts),
                validate=lambda _, x: x.isdigit() and int(x) > 0
            ),
            #inquirer.Text('system_prompt',
            #    message="System Prompt",
            #    default=state.system_prompt
            #),
        ])
        if result is None: return  # Handle prompt cancellation concisely
        state.num_attempts = int(result['num_attempts'])
        return MainMenu

class TargetLLMOptions:
    @classmethod
    def show(cls, state: AppConfig):
        models_list = get_langchain_chat_models_info().keys()
        print("Target LLM Options: Review and modify the target LLM configuration")
        print("------------------------------------------------------------------")
        
        # First get provider and model
        basic_result = inquirer.prompt([
            inquirer.List(
                'target_provider',
                message="LLM Provider configured in the AI chat application being fuzzed",
                choices=models_list,
                default=state.target_provider
            ),
            inquirer.Text('target_model',
                message="LLM Model configured in the AI chat application being fuzzed",
                default=state.target_model
            ),
        ])
        if basic_result is None: return  # Handle prompt cancellation concisely
        
        # Update state with basic settings
        state.target_provider = basic_result['target_provider']
        state.target_model = basic_result['target_model']
        
        # Ask for base URL if provider is ollama or open_ai
        if state.target_provider == 'ollama':
            base_url_result = inquirer.prompt([
                inquirer.Text('ollama_base_url',
                    message="Ollama Base URL (leave empty for default localhost:11434)",
                    default=state.ollama_base_url
                ),
            ])
            if base_url_result is not None:
                state.ollama_base_url = base_url_result['ollama_base_url']
        elif state.target_provider == 'open_ai':
            base_url_result = inquirer.prompt([
                inquirer.Text('openai_base_url',
                    message="OpenAI Base URL (leave empty for default api.openai.com)",
                    default=state.openai_base_url
                ),
            ])
            if base_url_result is not None:
                state.openai_base_url = base_url_result['openai_base_url']
        
        return MainMenu

class AttackLLMOptions:
    @classmethod
    def show(cls, state: AppConfig):
        models_list = get_langchain_chat_models_info().keys()
        print("Attack LLM Options: Review and modify the service LLM configuration used by the tool to help attack the system prompt")
        print("---------------------------------------------------------------------------------------------------------------------")
        
        # First get provider and model
        basic_result = inquirer.prompt([
            inquirer.List(
                'attack_provider',
                message="Service LLM Provider used to help attacking the system prompt",
                choices=models_list,
                default=state.attack_provider
            ),
            inquirer.Text('attack_model',
                message="Service LLM Model used to help attacking the system prompt",
                default=state.attack_model
            ),
        ])
        if basic_result is None: return  # Handle prompt cancellation concisely
        
        # Update state with basic settings
        state.attack_provider = basic_result['attack_provider']
        state.attack_model = basic_result['attack_model']
        
        # Ask for base URL if provider is ollama or open_ai
        if state.attack_provider == 'ollama':
            base_url_result = inquirer.prompt([
                inquirer.Text('ollama_base_url',
                    message="Ollama Base URL (leave empty for default localhost:11434)",
                    default=state.ollama_base_url
                ),
            ])
            if base_url_result is not None:
                state.ollama_base_url = base_url_result['ollama_base_url']
        elif state.attack_provider == 'open_ai':
            base_url_result = inquirer.prompt([
                inquirer.Text('openai_base_url',
                    message="OpenAI Base URL (leave empty for default api.openai.com)",
                    default=state.openai_base_url
                ),
            ])
            if base_url_result is not None:
                state.openai_base_url = base_url_result['openai_base_url']
        
        return MainMenu

class EmbeddingOptions:
    @classmethod
    def show(cls, state: AppConfig):
        print("Embedding Options: Review and modify the embedding provider configuration")
        print("--------------------------------------------------------------------------")
        
        # First get embedding provider and model
        basic_result = inquirer.prompt([
            inquirer.List(
                'embedding_provider',
                message="Embedding Provider for vector similarity search",
                choices=['ollama', 'open_ai'],
                default=state.embedding_provider
            ),
            inquirer.Text('embedding_model',
                message="Embedding Model for vector similarity search",
                default=state.embedding_model
            ),
        ])
        if basic_result is None: return  # Handle prompt cancellation concisely
        
        # Update state with basic settings
        state.embedding_provider = basic_result['embedding_provider']
        state.embedding_model = basic_result['embedding_model']
        
        # Ask for base URL based on selected provider
        if state.embedding_provider == 'ollama':
            # Use embedding-specific base URL if set, otherwise fall back to main ollama base URL
            default_url = state.embedding_ollama_base_url or state.ollama_base_url
            base_url_result = inquirer.prompt([
                inquirer.Text('embedding_ollama_base_url',
                    message="Ollama Embedding Base URL (leave empty to use main Ollama base URL or default localhost:11434)",
                    default=default_url
                ),
            ])
            if base_url_result is not None:
                state.embedding_ollama_base_url = base_url_result['embedding_ollama_base_url']
        elif state.embedding_provider == 'open_ai':
            # Use embedding-specific base URL if set, otherwise fall back to main openai base URL
            default_url = state.embedding_openai_base_url or state.openai_base_url
            base_url_result = inquirer.prompt([
                inquirer.Text('embedding_openai_base_url',
                    message="OpenAI Embedding Base URL (leave empty to use main OpenAI base URL or default api.openai.com)",
                    default=default_url
                ),
            ])
            if base_url_result is not None:
                state.embedding_openai_base_url = base_url_result['embedding_openai_base_url']
        
        return MainMenu

def interactive_shell(state: AppConfig):
    show_all_config(state)
    stage = MainMenu
    while stage:
        try:
            print()
            stage = stage.show(state)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            continue
        except ValueError as e:
            logger.warning(f"{colorama.Fore.RED}{colorama.Style.BRIGHT}Wrong value:{colorama.Style.RESET_ALL} {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            break
