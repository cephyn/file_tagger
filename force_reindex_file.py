    def force_reindex_file(self, file_path):
        """Force a file to be reindexed in the vector search database."""
        if not file_path or not os.path.isfile(file_path):
            QMessageBox.warning(self, "Error", f"The file {file_path} does not exist or is not a valid file.")
            return False
            
        try:
            # Get the file from database
            file_obj = self.db_session.query(File).filter_by(path=file_path).first()
            if not file_obj:
                QMessageBox.warning(self, "Error", f"The file {file_path} is not in the tag database.")
                return False
            
            print(f"\n=== Force reindexing file: {file_path} ===")
            
            # Check if AI is configured for summary generation
            ai_configured = False
            try:
                if self.config.get_selected_provider() and self.config.get_api_key(self.config.get_selected_provider()):
                    provider = self.config.get_selected_provider()
                    print(f"AI provider configured: {provider}")
                    ai_configured = True
                    
                    # Attach AI config information directly to db_session for use in VectorSearch
                    self.db_session.provider = provider
                    self.db_session.api_key = self.config.get_api_key(provider)
                    self.db_session.local_model_path = self.config.get_local_model_path()
                    self.db_session.local_model_type = self.config.get_local_model_type()
                else:
                    print("AI not configured - document summaries will not be generated")
            except Exception as e:
                print(f"Error checking AI configuration: {str(e)}")
            
            from vector_search.content_extractor import ContentExtractor
            
            # Create progress dialog for extraction
            progress = QProgressDialog("Extracting content...", "Cancel", 0, 0, self)
            progress.setWindowTitle("Extracting Content")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            progress.setCancelButton(None)  # No cancel button
            progress.setRange(0, 0)  # Show busy indicator (spinner)
            progress.show()
            
            # Variable to store extracted content
            extracted_content = None
            
            # Define callbacks for the async extraction
            def on_progress(message):
                progress.setLabelText(message)
                QApplication.processEvents()
            
            def on_extraction_complete(result):
                nonlocal extracted_content
                progress.close()
                
                if result.get('success', False):
                    extracted_content = result.get('content', '')
                    print(f"Content extraction completed successfully, length: {len(extracted_content)} characters")
                    finish_reindexing(extracted_content)
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"Content extraction failed: {error}")
                    QMessageBox.warning(
                        self, 
                        "Error", 
                        f"Content extraction failed: {error}"
                    )
            
            def finish_reindexing(content):
                if not content:
                    QMessageBox.warning(
                        self, 
                        "Error", 
                        f"Could not extract content from {os.path.basename(file_path)}.\n\n"
                        "This file type may not be supported for content extraction."
                    )
                    return False
                
                try:
                    # Remove existing entries before reindexing
                    print(f"Removing existing document entries for: {file_path}")
                    self.vector_search.remove_file(file_path)
                    
                    # Index the file with extracted content
                    print(f"Reindexing file: {file_path}")
                    self.vector_search.index_file(file_path, content)
                    
                    # Verify that the file was indexed with a summary
                    try:
                        results = self.vector_search.collection.get(
                            ids=[file_path], include=['metadatas']
                        )
                        if results and results['ids'] and len(results['ids']) > 0:
                            metadata = results['metadatas'][0]
                            print(f"Verification successful - file found in vector store")
                            if 'summary' in metadata and metadata['summary']:
                                print(f"Document summary generated: {metadata['summary']}")
                            else:
                                print("No document summary was generated")
                    except Exception as verify_err:
                        print(f"Error verifying file in vector store: {str(verify_err)}")
                    
                    QMessageBox.information(
                        self,
                        "Success",
                        f"The file {os.path.basename(file_path)} has been successfully reindexed."
                    )
                    return True
                except Exception as e:
                    print(f"Error during finishing reindexing: {str(e)}")
                    traceback.print_exc()
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Failed to complete reindexing: {str(e)}"
                    )
                    return False
            
            # Start the async extraction process
            print(f"Starting async content extraction from: {file_path}")
            ContentExtractor.extract_file_content_async(file_path, on_extraction_complete, on_progress)
            return True  # Return true immediately, the actual work happens asynchronously
            
        except Exception as e:
            print(f"Error during force reindex: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to reindex file: {str(e)}"
            )
            return False
