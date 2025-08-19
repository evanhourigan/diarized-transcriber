#!/usr/bin/env python3

import unittest
import tempfile
import os
from diarized_transcriber.markdown_exporter import format_timestamp, generate_markdown_transcript


class TestMarkdownExporter(unittest.TestCase):
    
    def test_format_timestamp(self):
        """Test timestamp formatting"""
        self.assertEqual(format_timestamp(0), "00:00")
        self.assertEqual(format_timestamp(61), "01:01")
        self.assertEqual(format_timestamp(125), "02:05")
        self.assertEqual(format_timestamp(3600), "60:00")
    
    def test_generate_markdown_with_speakers(self):
        """Test markdown generation with speaker diarization"""
        segments = [
            {"start": 0, "text": "Hello there", "speaker": "SPEAKER_1"},
            {"start": 2, "text": "Hi back", "speaker": "SPEAKER_2"},
            {"start": 4, "text": "How are you?", "speaker": "SPEAKER_1"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            generate_markdown_transcript(segments, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that speaker labels are present
            self.assertIn("**SPEAKER_1:**", content)
            self.assertIn("**SPEAKER_2:**", content)
            self.assertIn("Hello there", content)
            self.assertIn("Hi back", content)
            self.assertIn("How are you?", content)
            
        finally:
            os.unlink(temp_path)
    
    def test_generate_markdown_without_speakers(self):
        """Test markdown generation without speaker diarization"""
        segments = [
            {"start": 0, "text": "Hello there"},
            {"start": 2, "text": "Hi back"},
            {"start": 4, "text": "How are you?"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            generate_markdown_transcript(segments, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that no speaker labels are present
            self.assertNotIn("**SPEAKER", content)
            self.assertIn("Hello there", content)
            self.assertIn("Hi back", content)
            self.assertIn("How are you?", content)
            
        finally:
            os.unlink(temp_path)
    
    def test_generate_markdown_with_timestamps(self):
        """Test markdown generation with timestamps enabled"""
        segments = [
            {"start": 0, "text": "Hello there", "speaker": "SPEAKER_1"},
            {"start": 61, "text": "Hi back", "speaker": "SPEAKER_2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            generate_markdown_transcript(segments, temp_path, include_timestamps=True)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that timestamps are present
            self.assertIn("[00:00]", content)
            self.assertIn("[01:01]", content)
            
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
