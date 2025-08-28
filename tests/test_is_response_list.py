import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock
from ps_fuzz.util import is_response_list
from ps_fuzz.app_config import AppConfig
from ps_fuzz.prompt_injection_fuzzer import _build_client_kwargs, _build_embedding_config, isSkipped
from ps_fuzz.test_base import TestStatus

def test_min_items_count_zero():
    # Test behavior when min_items_count is set to 0
    response = "item1; item2; item3"
    expected = False
    assert is_response_list(response, 0) == expected, "Function should return False when min_items_count is 0"

    # Testing empty input for completeness
    response_empty = ""
    assert is_response_list(response_empty, 0) == expected, "Function should return False when min_items_count is 0 even for empty input"

# Test cases for semicolon-separated lists
@pytest.mark.parametrize("response, expected", [
    ("item1; item2; item3", True),  # three items, correct for minimum of 3
    ("item1; item2", False),  # two items, not enough for minimum of 3
    ("item1; item2; item3;", True),  # extra trailing semicolon, still three items
    ("item1 item2 item3", False),  # no separators, not recognized as a list
    ("", False),  # empty string, no items at all
    ("item1;", False)  # only one item, not enough for minimum of 3
])
def test_semicolon_lists(response, expected):
    assert is_response_list(response, 3) == expected

# Test cases for simple list markers (bulleted lists)
@pytest.mark.parametrize("response, expected", [
    ("- item1\n- item2\n- item3", True),  # correct number of bulleted items
    ("* item1\n* item2\n* item3", True),  # correct using asterisk bullets
    ("• item1\n• item2\n• item3", True),  # correct using bullet symbol
    ("- item1\n- item2", False),  # not enough items
    ("* item1", False),  # only one item with asterisk
    ("item1\nitem2\nitem3", False),  # no markers, just new lines
    ("-", False)  # single dash, not a list
])
def test_simple_list_markers(response, expected):
    assert is_response_list(response, 3) == expected

# Test cases for complex list markers (numbered or lettered lists)
@pytest.mark.parametrize("response, expected", [
    ("1. item1\n2. item2\n3. item3", True),  # correctly formatted numbered list
    ("a. item1\nb. item2\nc. item3", True),  # correctly formatted lettered list
    ("A. item1\nB. item2\nC. item3", True),  # uppercase lettered list
    ("1. item1\n2. item2", False),  # not enough numbered items
    ("a. item1\nb. ", False),  # incomplete second item in a lettered list
    ("1. item1 2. item2 3. item3", False),  # items not on new lines
    ("1.item1", False)  # only one item without proper formatting
])
def test_complex_list_markers(response, expected):
    assert is_response_list(response, 3) == expected

@pytest.mark.parametrize("response, expected", [
    ("step 1. Prepare; step 2. Mix; step 3. Bake", True),  # correctly formatted with introductory words and semicolons
    ("phase 1: Research; phase 2: Design; phase 3: Implement", True),  # phases with colons and semicolons
    ("part a. Start; part b. Continue; part c. End", True),  # parts with letters and semicolons
    ("item 1- Collect data; item 2- Analyze data; item 3- Report findings", True),  # items with dashes and semicolons
    ("1. step Gather; 2. step Compile; 3. step Report", True),  # numbered steps without new lines but with clear steps
    ("note 1. Overview\nnote 2. Details\nnote 3. Summary", True),  # notes with new lines
])
def test_introductory_words_in_lists(response, expected):
    assert is_response_list(response, 3) == expected


class TestAppConfigEmbeddingProperties:
    """Test class for AppConfig embedding-related properties."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_file = f.name
        yield temp_file
        os.unlink(temp_file)
    
    @pytest.fixture
    def mock_config_data(self):
        """Mock configuration data for testing."""
        return {
            'attack_provider': 'open_ai',
            'attack_model': 'gpt-3.5-turbo',
            'target_provider': 'open_ai',
            'target_model': 'gpt-3.5-turbo',
            'num_attempts': 3,
            'num_threads': 4,
            'attack_temperature': 0.6,
            'system_prompt': '',
            'custom_benchmark': '',
            'tests': [],
            'embedding_provider': 'ollama',
            'embedding_model': 'nomic-embed-text',
            'embedding_ollama_base_url': 'http://localhost:11434',
            'embedding_openai_base_url': 'https://api.openai.com/v1'
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_embedding_provider_getter_setter_valid(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test valid embedding providers ('ollama', 'open_ai')."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'embedding_provider': 'ollama'}
        
        config = AppConfig('test_config.json')
        
        # Test getter
        assert config.embedding_provider == 'ollama'
        
        # Test setter with valid values
        config.embedding_provider = 'open_ai'
        assert config.config_state['embedding_provider'] == 'open_ai'
        mock_json_dump.assert_called()
        
        config.embedding_provider = 'ollama'
        assert config.config_state['embedding_provider'] == 'ollama'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_embedding_provider_setter_empty_raises_error(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test ValueError on empty embedding provider."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'embedding_provider': 'ollama'}
        
        config = AppConfig('test_config.json')
        
        with pytest.raises(ValueError, match="Embedding provider cannot be empty"):
            config.embedding_provider = ''
        
        with pytest.raises(ValueError, match="Embedding provider cannot be empty"):
            config.embedding_provider = None
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_embedding_model_getter_setter_valid(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test valid embedding model names."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'embedding_model': 'nomic-embed-text'}
        
        config = AppConfig('test_config.json')
        
        # Test getter
        assert config.embedding_model == 'nomic-embed-text'
        
        # Test setter with valid values
        config.embedding_model = 'text-embedding-ada-002'
        assert config.config_state['embedding_model'] == 'text-embedding-ada-002'
        mock_json_dump.assert_called()
        
        config.embedding_model = 'all-MiniLM-L6-v2'
        assert config.config_state['embedding_model'] == 'all-MiniLM-L6-v2'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_embedding_model_setter_empty_raises_error(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test ValueError on empty embedding model."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'embedding_model': 'nomic-embed-text'}
        
        config = AppConfig('test_config.json')
        
        with pytest.raises(ValueError, match="Embedding model cannot be empty"):
            config.embedding_model = ''
        
        with pytest.raises(ValueError, match="Embedding model cannot be empty"):
            config.embedding_model = None
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_embedding_ollama_base_url_getter_setter(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test embedding Ollama base URL setting/getting (allows empty)."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'embedding_ollama_base_url': 'http://localhost:11434'}
        
        config = AppConfig('test_config.json')
        
        # Test getter
        assert config.embedding_ollama_base_url == 'http://localhost:11434'
        
        # Test setter with valid URL
        config.embedding_ollama_base_url = 'http://custom-ollama:8080'
        assert config.config_state['embedding_ollama_base_url'] == 'http://custom-ollama:8080'
        mock_json_dump.assert_called()
        
        # Test setter with empty value (should be allowed)
        config.embedding_ollama_base_url = ''
        assert config.config_state['embedding_ollama_base_url'] == ''
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_embedding_openai_base_url_getter_setter(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test embedding OpenAI base URL setting/getting (allows empty)."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'embedding_openai_base_url': 'https://api.openai.com/v1'}
        
        config = AppConfig('test_config.json')
        
        # Test getter
        assert config.embedding_openai_base_url == 'https://api.openai.com/v1'
        
        # Test setter with valid URL
        config.embedding_openai_base_url = 'https://custom-openai.example.com/v1'
        assert config.config_state['embedding_openai_base_url'] == 'https://custom-openai.example.com/v1'
        mock_json_dump.assert_called()
        
        # Test setter with empty value (should be allowed)
        config.embedding_openai_base_url = ''
        assert config.config_state['embedding_openai_base_url'] == ''
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_embedding_properties_persistence(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test embedding properties config save/load cycle."""
        mock_exists.return_value = True
        initial_config = {
            'embedding_provider': 'ollama',
            'embedding_model': 'nomic-embed-text',
            'embedding_ollama_base_url': 'http://localhost:11434',
            'embedding_openai_base_url': ''
        }
        mock_json_load.return_value = initial_config
        
        config = AppConfig('test_config.json')
        
        # Modify properties
        config.embedding_provider = 'open_ai'
        config.embedding_model = 'text-embedding-ada-002'
        config.embedding_ollama_base_url = ''
        config.embedding_openai_base_url = 'https://api.openai.com/v1'
        
        # Verify save was called for each property change
        assert mock_json_dump.call_count == 4
        
        # Verify final state
        assert config.config_state['embedding_provider'] == 'open_ai'
        assert config.config_state['embedding_model'] == 'text-embedding-ada-002'
        assert config.config_state['embedding_ollama_base_url'] == ''
        assert config.config_state['embedding_openai_base_url'] == 'https://api.openai.com/v1'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_embedding_properties_defaults(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test embedding properties default empty values."""
        mock_exists.return_value = True
        mock_json_load.return_value = {}  # Empty config
        
        config = AppConfig('test_config.json')
        
        # Test default values (should be empty strings)
        assert config.embedding_provider == ''
        assert config.embedding_model == ''
        assert config.embedding_ollama_base_url == ''
        assert config.embedding_openai_base_url == ''


class TestAppConfigBaseURLProperties:
    """Test class for AppConfig base URL properties."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_file = f.name
        yield temp_file
        os.unlink(temp_file)
    
    @pytest.fixture
    def mock_config_data(self):
        """Mock configuration data for testing."""
        return {
            'attack_provider': 'open_ai',
            'attack_model': 'gpt-3.5-turbo',
            'target_provider': 'open_ai',
            'target_model': 'gpt-3.5-turbo',
            'num_attempts': 3,
            'num_threads': 4,
            'attack_temperature': 0.6,
            'system_prompt': '',
            'custom_benchmark': '',
            'tests': [],
            'ollama_base_url': 'http://localhost:11434',
            'openai_base_url': 'https://api.openai.com/v1'
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_ollama_base_url_getter_setter(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test Ollama base URL setting/getting."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'ollama_base_url': 'http://localhost:11434'}
        
        config = AppConfig('test_config.json')
        
        # Test getter
        assert config.ollama_base_url == 'http://localhost:11434'
        
        # Test setter with valid URL
        config.ollama_base_url = 'http://custom-ollama:8080'
        assert config.config_state['ollama_base_url'] == 'http://custom-ollama:8080'
        mock_json_dump.assert_called()
        
        # Test setter with empty value (should be allowed)
        config.ollama_base_url = ''
        assert config.config_state['ollama_base_url'] == ''
        
        # Test setter with different protocols
        config.ollama_base_url = 'https://secure-ollama.example.com'
        assert config.config_state['ollama_base_url'] == 'https://secure-ollama.example.com'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_openai_base_url_getter_setter(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test OpenAI base URL setting/getting."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'openai_base_url': 'https://api.openai.com/v1'}
        
        config = AppConfig('test_config.json')
        
        # Test getter
        assert config.openai_base_url == 'https://api.openai.com/v1'
        
        # Test setter with valid URL
        config.openai_base_url = 'https://custom-openai.example.com/v1'
        assert config.config_state['openai_base_url'] == 'https://custom-openai.example.com/v1'
        mock_json_dump.assert_called()
        
        # Test setter with empty value (should be allowed)
        config.openai_base_url = ''
        assert config.config_state['openai_base_url'] == ''
        
        # Test setter with Azure OpenAI format
        config.openai_base_url = 'https://myresource.openai.azure.com/'
        assert config.config_state['openai_base_url'] == 'https://myresource.openai.azure.com/'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_base_url_properties_persistence(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test base URL properties config save/load cycle."""
        mock_exists.return_value = True
        initial_config = {
            'ollama_base_url': 'http://localhost:11434',
            'openai_base_url': 'https://api.openai.com/v1'
        }
        mock_json_load.return_value = initial_config
        
        config = AppConfig('test_config.json')
        
        # Modify properties
        config.ollama_base_url = 'http://custom-ollama:8080'
        config.openai_base_url = 'https://custom-openai.example.com/v1'
        
        # Verify save was called for each property change
        assert mock_json_dump.call_count == 2
        
        # Verify final state
        assert config.config_state['ollama_base_url'] == 'http://custom-ollama:8080'
        assert config.config_state['openai_base_url'] == 'https://custom-openai.example.com/v1'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_base_url_properties_defaults(self, mock_exists, mock_json_load, mock_json_dump, mock_file):
        """Test base URL properties default empty values."""
        mock_exists.return_value = True
        mock_json_load.return_value = {}  # Empty config
        
        config = AppConfig('test_config.json')
        
        # Test default values (should be empty strings)
        assert config.ollama_base_url == ''
        assert config.openai_base_url == ''
    
    @pytest.mark.parametrize("url_property,test_urls", [
        ('ollama_base_url', [
            'http://localhost:11434',
            'https://ollama.example.com',
            'http://192.168.1.100:8080',
            'https://secure-ollama.company.com:443'
        ]),
        ('openai_base_url', [
            'https://api.openai.com/v1',
            'https://custom-openai.example.com/v1',
            'https://myresource.openai.azure.com/',
            'http://localhost:8000/v1'
        ])
    ])
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_base_url_various_formats(self, mock_exists, mock_json_load, mock_json_dump, mock_file, url_property, test_urls):
        """Test base URL properties with various URL formats."""
        mock_exists.return_value = True
        mock_json_load.return_value = {}
        
        config = AppConfig('test_config.json')
        
        for url in test_urls:
            setattr(config, url_property, url)
            assert config.config_state[url_property] == url
            assert getattr(config, url_property) == url



class TestHelperFunctions:
    """Test class for helper functions from prompt_injection_fuzzer.py."""
    
    def test_build_client_kwargs_ollama_with_base_url(self):
        """Test kwargs building for Ollama with base URL."""
        # Create mock AppConfig with ollama_base_url
        mock_app_config = MagicMock()
        mock_app_config.ollama_base_url = 'http://localhost:11434'
        
        result = _build_client_kwargs(mock_app_config, 'ollama', 'llama2', 0.7)
        
        expected = {
            'model': 'llama2',
            'temperature': 0.7,
            'ollama_base_url': 'http://localhost:11434'
        }
        assert result == expected
    
    def test_build_client_kwargs_ollama_without_base_url(self):
        """Test kwargs building for Ollama without base URL."""
        # Create mock AppConfig without ollama_base_url
        mock_app_config = MagicMock()
        mock_app_config.ollama_base_url = ''
        
        result = _build_client_kwargs(mock_app_config, 'ollama', 'llama2', 0.7)
        
        expected = {
            'model': 'llama2',
            'temperature': 0.7
        }
        assert result == expected
    
    def test_build_client_kwargs_ollama_missing_attribute(self):
        """Test kwargs building for Ollama when base URL attribute is missing."""
        # Create mock AppConfig without ollama_base_url attribute
        mock_app_config = MagicMock()
        del mock_app_config.ollama_base_url  # Remove the attribute
        
        result = _build_client_kwargs(mock_app_config, 'ollama', 'llama2', 0.7)
        
        expected = {
            'model': 'llama2',
            'temperature': 0.7
        }
        assert result == expected
    
    def test_build_client_kwargs_openai_with_base_url(self):
        """Test kwargs building for OpenAI with base URL."""
        # Create mock AppConfig with openai_base_url
        mock_app_config = MagicMock()
        mock_app_config.openai_base_url = 'https://api.openai.com/v1'
        
        result = _build_client_kwargs(mock_app_config, 'open_ai', 'gpt-3.5-turbo', 0.5)
        
        expected = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.5,
            'openai_base_url': 'https://api.openai.com/v1'
        }
        assert result == expected
    
    def test_build_client_kwargs_openai_without_base_url(self):
        """Test kwargs building for OpenAI without base URL."""
        # Create mock AppConfig without openai_base_url
        mock_app_config = MagicMock()
        mock_app_config.openai_base_url = ''
        
        result = _build_client_kwargs(mock_app_config, 'open_ai', 'gpt-3.5-turbo', 0.5)
        
        expected = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.5
        }
        assert result == expected
    
    def test_build_client_kwargs_openai_missing_attribute(self):
        """Test kwargs building for OpenAI when base URL attribute is missing."""
        # Create mock AppConfig without openai_base_url attribute
        mock_app_config = MagicMock()
        del mock_app_config.openai_base_url  # Remove the attribute
        
        result = _build_client_kwargs(mock_app_config, 'open_ai', 'gpt-3.5-turbo', 0.5)
        
        expected = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.5
        }
        assert result == expected
    
    def test_build_client_kwargs_other_providers(self):
        """Test kwargs building for other providers."""
        # Create mock AppConfig with various base URLs
        mock_app_config = MagicMock()
        mock_app_config.ollama_base_url = 'http://localhost:11434'
        mock_app_config.openai_base_url = 'https://api.openai.com/v1'
        
        # Test with anthropic provider (should not include any base URLs)
        result = _build_client_kwargs(mock_app_config, 'anthropic', 'claude-3-sonnet', 0.3)
        
        expected = {
            'model': 'claude-3-sonnet',
            'temperature': 0.3
        }
        assert result == expected
        
        # Test with google provider (should not include any base URLs)
        result = _build_client_kwargs(mock_app_config, 'google', 'gemini-pro', 0.8)
        
        expected = {
            'model': 'gemini-pro',
            'temperature': 0.8
        }
        assert result == expected
    
    def test_build_embedding_config_complete(self):
        """Test embedding config with all properties."""
        # Create mock AppConfig with all embedding properties
        mock_app_config = MagicMock()
        mock_app_config.embedding_provider = 'ollama'
        mock_app_config.embedding_model = 'nomic-embed-text'
        mock_app_config.embedding_ollama_base_url = 'http://localhost:11434'
        mock_app_config.embedding_openai_base_url = 'https://api.openai.com/v1'
        
        result = _build_embedding_config(mock_app_config)
        
        expected = {
            'embedding_provider': 'ollama',
            'embedding_model': 'nomic-embed-text',
            'embedding_ollama_base_url': 'http://localhost:11434',
            'embedding_openai_base_url': 'https://api.openai.com/v1'
        }
        assert result == expected
    
    def test_build_embedding_config_partial(self):
        """Test embedding config with missing properties."""
        # Create mock AppConfig with some missing embedding properties
        mock_app_config = MagicMock()
        mock_app_config.embedding_provider = 'open_ai'
        mock_app_config.embedding_model = 'text-embedding-ada-002'
        # Missing embedding_ollama_base_url and embedding_openai_base_url attributes
        del mock_app_config.embedding_ollama_base_url
        del mock_app_config.embedding_openai_base_url
        
        result = _build_embedding_config(mock_app_config)
        
        expected = {
            'embedding_provider': 'open_ai',
            'embedding_model': 'text-embedding-ada-002',
            'embedding_ollama_base_url': '',  # Default empty string
            'embedding_openai_base_url': ''   # Default empty string
        }
        assert result == expected
    
    def test_build_embedding_config_empty(self):
        """Test embedding config with empty AppConfig."""
        # Create mock AppConfig with no embedding attributes
        mock_app_config = MagicMock()
        del mock_app_config.embedding_provider
        del mock_app_config.embedding_model
        del mock_app_config.embedding_ollama_base_url
        del mock_app_config.embedding_openai_base_url
        
        result = _build_embedding_config(mock_app_config)
        
        expected = {
            'embedding_provider': '',
            'embedding_model': '',
            'embedding_ollama_base_url': '',
            'embedding_openai_base_url': ''
        }
        assert result == expected
    
    @pytest.mark.parametrize("provider,base_url_attr,base_url_value,expected_key", [
        ('ollama', 'ollama_base_url', 'http://localhost:11434', 'ollama_base_url'),
        ('ollama', 'ollama_base_url', 'http://custom-ollama:8080', 'ollama_base_url'),
        ('open_ai', 'openai_base_url', 'https://api.openai.com/v1', 'openai_base_url'),
        ('open_ai', 'openai_base_url', 'https://custom-openai.example.com/v1', 'openai_base_url'),
    ])
    def test_build_client_kwargs_parametrized(self, provider, base_url_attr, base_url_value, expected_key):
        """Test kwargs building with parametrized provider and base URL combinations."""
        mock_app_config = MagicMock()
        setattr(mock_app_config, base_url_attr, base_url_value)
        
        result = _build_client_kwargs(mock_app_config, provider, 'test-model', 0.6)
        
        expected = {
            'model': 'test-model',
            'temperature': 0.6,
            expected_key: base_url_value
        }
        assert result == expected
    
    @pytest.mark.parametrize("embedding_provider,embedding_model,ollama_url,openai_url", [
        ('ollama', 'nomic-embed-text', 'http://localhost:11434', ''),
        ('open_ai', 'text-embedding-ada-002', '', 'https://api.openai.com/v1'),
        ('ollama', 'all-MiniLM-L6-v2', 'http://custom-ollama:8080', 'https://custom-openai.com/v1'),
        ('', '', '', ''),  # Empty configuration
    ])
    def test_build_embedding_config_parametrized(self, embedding_provider, embedding_model, ollama_url, openai_url):
        """Test embedding config building with parametrized values."""
        mock_app_config = MagicMock()
        mock_app_config.embedding_provider = embedding_provider
        mock_app_config.embedding_model = embedding_model
        mock_app_config.embedding_ollama_base_url = ollama_url
        mock_app_config.embedding_openai_base_url = openai_url
        
        result = _build_embedding_config(mock_app_config)
        
        expected = {
            'embedding_provider': embedding_provider,
            'embedding_model': embedding_model,
            'embedding_ollama_base_url': ollama_url,
            'embedding_openai_base_url': openai_url
        }
        assert result == expected


class TestStatusSkippedFunctionality:
    """Test class for TestStatus skipped functionality."""
    
    def test_skipped_count_property(self):
        """Test skipped_count property getter."""
        status = TestStatus()
        
        # Test initial value
        assert status.skipped_count == 0
        
        # Test after manual increment (simulating internal behavior)
        status.skipped_count = 5
        assert status.skipped_count == 5
    
    def test_report_skipped_increments_count(self):
        """Test report_skipped() increments skipped_count."""
        status = TestStatus()
        
        # Initial state
        assert status.skipped_count == 0
        assert status.total_count == 0
        
        # Report one skipped test
        status.report_skipped("test prompt", "Test skipped due to missing config")
        assert status.skipped_count == 1
        assert status.total_count == 1
        
        # Report another skipped test
        status.report_skipped("another prompt", "Another skip reason")
        assert status.skipped_count == 2
        assert status.total_count == 2
    
    def test_report_skipped_adds_log_entry(self):
        """Test report_skipped() adds proper log entry."""
        status = TestStatus()
        
        prompt = "test prompt for skipping"
        additional_info = "Custom skip reason"
        
        status.report_skipped(prompt, additional_info)
        
        # Check log entry was added
        assert len(status.log) == 1
        log_entry = status.log[0]
        
        # Verify log entry properties
        assert log_entry.prompt == prompt
        assert log_entry.response is None  # Skipped tests have no response
        assert log_entry.success is False  # Skipped tests are not successful
        assert log_entry.additional_info == additional_info
    
    def test_report_skipped_updates_total_count(self):
        """Test report_skipped() increments total_count."""
        status = TestStatus()
        
        # Initial state
        assert status.total_count == 0
        
        # Report skipped test
        status.report_skipped("test prompt")
        assert status.total_count == 1
        
        # Report another type of result to verify total_count continues incrementing
        status.report_breach("breach prompt", "breach response")
        assert status.total_count == 2
        assert status.skipped_count == 1  # Should remain 1
        assert status.breach_count == 1
    
    def test_report_skipped_custom_message(self):
        """Test report_skipped() with custom additional_info parameter."""
        status = TestStatus()
        
        # Test with default message
        status.report_skipped("prompt1")
        assert status.log[0].additional_info == "Test skipped"
        
        # Test with custom message
        custom_message = "Skipped due to missing API key"
        status.report_skipped("prompt2", custom_message)
        assert status.log[1].additional_info == custom_message
    
    def test_multiple_skipped_reports(self):
        """Test multiple skipped reports accumulate correctly."""
        status = TestStatus()
        
        # Report multiple skipped tests
        for i in range(5):
            status.report_skipped(f"prompt_{i}", f"Skip reason {i}")
        
        # Verify counts
        assert status.skipped_count == 5
        assert status.total_count == 5
        assert len(status.log) == 5
        
        # Verify all log entries
        for i, log_entry in enumerate(status.log):
            assert log_entry.prompt == f"prompt_{i}"
            assert log_entry.additional_info == f"Skip reason {i}"
            assert log_entry.response is None
            assert log_entry.success is False
    
    def test_str_method_includes_skipped_count(self):
        """Test __str__() method includes skipped_count in representation."""
        status = TestStatus()
        
        # Test with no skipped tests
        str_repr = str(status)
        assert "skipped_count=0" in str_repr
        
        # Add some skipped tests
        status.report_skipped("prompt1")
        status.report_skipped("prompt2")
        
        str_repr = str(status)
        assert "skipped_count=2" in str_repr
        assert "total_count=2" in str_repr
        
        # Verify full format
        expected_parts = [
            "TestStatus(",
            "breach_count=0",
            "resilient_count=0", 
            "skipped_count=2",
            "total_count=2",
            "log:2 entries"
        ]
        for part in expected_parts:
            assert part in str_repr


class TestIsSkippedFunction:
    """Test class for isSkipped function from prompt_injection_fuzzer.py."""
    
    def test_is_skipped_only_skipped(self):
        """Test isSkipped returns True when only skipped_count > 0."""
        status = TestStatus()
        
        # Initially should be False (no results)
        assert isSkipped(status) is False
        
        # Add only skipped results
        status.report_skipped("prompt1")
        assert isSkipped(status) is True
        
        # Add more skipped results
        status.report_skipped("prompt2")
        assert isSkipped(status) is True
    
    def test_is_skipped_with_breaches(self):
        """Test isSkipped returns False when has breaches."""
        status = TestStatus()
        
        # Add skipped and breach results
        status.report_skipped("skipped_prompt")
        status.report_breach("breach_prompt", "breach_response")
        
        assert isSkipped(status) is False
        assert status.skipped_count > 0
        assert status.breach_count > 0
    
    def test_is_skipped_with_resilient(self):
        """Test isSkipped returns False when has resilient count."""
        status = TestStatus()
        
        # Add skipped and resilient results
        status.report_skipped("skipped_prompt")
        status.report_resilient("resilient_prompt", "resilient_response")
        
        assert isSkipped(status) is False
        assert status.skipped_count > 0
        assert status.resilient_count > 0
    
    def test_is_skipped_with_errors(self):
        """Test isSkipped returns False when has errors."""
        status = TestStatus()
        
        # Add skipped and error results
        status.report_skipped("skipped_prompt")
        status.report_error("error_prompt", "Error occurred")
        
        assert isSkipped(status) is False
        assert status.skipped_count > 0
        assert status.error_count > 0
    
    def test_is_skipped_mixed_results(self):
        """Test isSkipped returns False with mixed results."""
        status = TestStatus()
        
        # Add various types of results
        status.report_skipped("skipped_prompt")
        status.report_breach("breach_prompt", "breach_response")
        status.report_resilient("resilient_prompt", "resilient_response")
        status.report_error("error_prompt", "Error occurred")
        
        assert isSkipped(status) is False
        assert status.skipped_count > 0
        assert status.breach_count > 0
        assert status.resilient_count > 0
        assert status.error_count > 0
    
    def test_is_skipped_no_results(self):
        """Test isSkipped returns False with no results."""
        status = TestStatus()
        
        # Empty status should return False
        assert isSkipped(status) is False
        assert status.skipped_count == 0
        assert status.breach_count == 0
        assert status.resilient_count == 0
        assert status.error_count == 0
    
    @pytest.mark.parametrize("breach_count,resilient_count,error_count,skipped_count,expected", [
        (0, 0, 0, 0, False),  # No results
        (0, 0, 0, 1, True),   # Only skipped
        (0, 0, 0, 5, True),   # Multiple skipped only
        (1, 0, 0, 1, False),  # Skipped + breach
        (0, 1, 0, 1, False),  # Skipped + resilient
        (0, 0, 1, 1, False),  # Skipped + error
        (1, 1, 1, 1, False),  # All types
        (2, 0, 0, 0, False),  # Only breaches
        (0, 3, 0, 0, False),  # Only resilient
        (0, 0, 4, 0, False),  # Only errors
    ])
    def test_is_skipped_parametrized(self, breach_count, resilient_count, error_count, skipped_count, expected):
        """Test isSkipped function with parametrized test status configurations."""
        status = TestStatus()
        
        # Set counts directly to test the logic
        status.breach_count = breach_count
        status.resilient_count = resilient_count
        status.error_count = error_count
        status.skipped_count = skipped_count
        
        assert isSkipped(status) is expected