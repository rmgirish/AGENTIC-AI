CREATE TRIGGER IF NOT EXISTS message_no_update
BEFORE UPDATE ON message
BEGIN
    SELECT RAISE(FAIL, 'message is append-only');
END;

CREATE TRIGGER IF NOT EXISTS message_no_delete
BEFORE DELETE ON message
BEGIN
    SELECT RAISE(FAIL, 'message is append-only');
END;
