#!/usr/bin/env python3

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from diarized_transcriber.diarization import run_transcribe_with_diarization


class TestDiarization(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_audio_path = "test_audio.wav"
        self.test_output_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_output_dir)
    
    @patch('torch.cuda.is_available')
    @patch('whisperx.load_model')
    @patch('whisperx.load_align_model')
    @patch('whisperx.align')
    @patch('whisperx.diarize.DiarizationPipeline')
    @patch('whisperx.diarize.assign_word_speakers')
    def test_run_transcribe_with_diarization_full_pipeline(self, mock_assign_speakers, mock_diarize_pipeline, 
                                                          mock_align, mock_load_align, mock_load_model, mock_cuda):
        """Test the full transcription and diarization pipeline"""
        # Mock CUDA availability
        mock_cuda.return_value = False
        
        # Mock Whisper model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
            'language': 'en'
        }
        mock_load_model.return_value = mock_model
        
        # Mock alignment model
        mock_align_model = MagicMock()
        mock_align_metadata = {'language': 'en'}
        mock_load_align.return_value = (mock_align_model, mock_align_metadata)
        
        # Mock alignment result
        mock_align.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
            'language': 'en'
        }
        
        # Mock diarization pipeline
        mock_diarize_pipe = MagicMock()
        mock_diarize_pipe.return_value = [{'start': 0, 'end': 5, 'speaker': 'SPEAKER_1'}]
        mock_diarize_pipeline.return_value = mock_diarize_pipe
        
        # Mock speaker assignment
        mock_assign_speakers.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Hello world', 'speaker': 'SPEAKER_1'}],
            'language': 'en'
        }
        
        # Set environment variable for testing
        with patch.dict(os.environ, {'HUGGINGFACE_TOKEN': 'test_token'}):
            result = run_transcribe_with_diarization(
                self.test_audio_path, 
                self.test_output_dir, 
                model_size="medium",
                skip_diarization=False,
                num_speakers=2,
                quiet=True
            )
        
        # Verify the pipeline was called correctly
        mock_load_model.assert_called_once_with("medium", "cpu", compute_type="float32")
        mock_model.transcribe.assert_called_once_with(self.test_audio_path)
        mock_load_align.assert_called_once_with("en", "cpu")
        mock_align.assert_called_once()
        mock_diarize_pipeline.assert_called_once()
        mock_assign_speakers.assert_called_once()
        
        # Verify result structure
        self.assertIn('segments', result)
        self.assertIn('language', result)
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(result['segments'][0]['speaker'], 'SPEAKER_1')
    
    @patch('torch.cuda.is_available')
    @patch('whisperx.load_model')
    @patch('whisperx.load_align_model')
    @patch('whisperx.align')
    def test_run_transcribe_skip_diarization(self, mock_align, mock_load_align, mock_load_model, mock_cuda):
        """Test transcription without diarization"""
        # Mock CUDA availability
        mock_cuda.return_value = False
        
        # Mock Whisper model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
            'language': 'en'
        }
        mock_load_model.return_value = mock_model
        
        # Mock alignment model
        mock_align_model = MagicMock()
        mock_align_metadata = {'language': 'en'}
        mock_load_align.return_value = (mock_align_model, mock_align_metadata)
        
        # Mock alignment result
        mock_align.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
            'language': 'en'
        }
        
        result = run_transcribe_with_diarization(
            self.test_audio_path, 
            self.test_output_dir, 
            model_size="base",
            skip_diarization=True,
            quiet=True
        )
        
        # Verify diarization was skipped
        self.assertIn('segments', result)
        self.assertIn('language', result)
        # Should not have speaker information
        self.assertNotIn('speaker', result['segments'][0])
    
    @patch('torch.cuda.is_available')
    @patch('whisperx.load_model')
    @patch('whisperx.load_align_model')
    @patch('whisperx.align')
    @patch('whisperx.diarize.DiarizationPipeline')
    @patch('whisperx.diarize.assign_word_speakers')
    def test_run_transcribe_no_huggingface_token(self, mock_assign_speakers, mock_diarize_pipeline, 
                                                mock_align, mock_load_align, mock_load_model, mock_cuda):
        """Test behavior when HUGGINGFACE_TOKEN is not set"""
        # Mock CUDA availability
        mock_cuda.return_value = False
        
        # Mock Whisper model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
            'language': 'en'
        }
        mock_load_model.return_value = mock_model
        
        # Mock alignment model
        mock_align_model = MagicMock()
        mock_align_metadata = {'language': 'en'}
        mock_load_align.return_value = (mock_align_model, mock_align_metadata)
        
        # Mock alignment result
        mock_align.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
            'language': 'en'
        }
        
        # Ensure no token is set
        with patch.dict(os.environ, {}, clear=True):
            result = run_transcribe_with_diarization(
                self.test_audio_path, 
                self.test_output_dir, 
                model_size="medium",
                skip_diarization=False,
                quiet=True
            )
        
        # Should complete without diarization
        self.assertIn('segments', result)
        self.assertIn('language', result)
        # Should not have speaker information
        self.assertNotIn('speaker', result['segments'][0])
    
    @patch('torch.cuda.is_available')
    def test_device_selection(self, mock_cuda):
        """Test device selection logic"""
        # Test CPU selection
        mock_cuda.return_value = False
        
        with patch('whisperx.load_model') as mock_load_model, \
             patch('whisperx.load_align_model') as mock_load_align, \
             patch('whisperx.align') as mock_align:
            
            # Mock all the necessary components
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
                'language': 'en'
            }
            mock_load_model.return_value = mock_model
            
            mock_align_model = MagicMock()
            mock_align_metadata = {'language': 'en'}
            mock_load_align.return_value = (mock_align_model, mock_align_metadata)
            
            mock_align.return_value = {
                'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
                'language': 'en'
            }
            
            run_transcribe_with_diarization(
                self.test_audio_path, 
                self.test_output_dir, 
                skip_diarization=True,
                quiet=True
            )
            
            # Verify CPU device was used
            mock_load_model.assert_called_once_with("large-v3", "cpu", compute_type="float32")
            mock_load_align.assert_called_once_with("en", "cpu")


if __name__ == '__main__':
    unittest.main()
