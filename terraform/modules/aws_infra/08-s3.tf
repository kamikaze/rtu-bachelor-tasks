# S3 buckets

resource "aws_s3_bucket" "loki_develop" {
  bucket = "${var.project}-loki-develop"

  tags = {
    Name         = "DEV Loki log storage",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_s3_bucket_public_access_block" "loki_develop_pab" {
  bucket = aws_s3_bucket.loki_develop.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "loki_main" {
  bucket = "${var.project}-loki-main"

  tags = {
    Name         = "PROD Loki log storage",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_s3_bucket_public_access_block" "loki_main_pab" {
  bucket = aws_s3_bucket.loki_main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket" "uploads" {
  bucket = "${var.project}-uploads"

  tags = {
    Name         = "Uploaded documents",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_s3_bucket_public_access_block" "uploads_pab" {
  bucket = aws_s3_bucket.uploads.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "reports" {
  bucket = "${var.project}-reports"

  tags = {
    Name         = "Reports",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_s3_bucket" "develop" {
  bucket = "${var.project}-develop"

  tags = {
    Name         = "DEV objects",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_s3_bucket_public_access_block" "develop_pab" {
  bucket = aws_s3_bucket.develop.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "main" {
  bucket = "${var.project}-main"

  tags = {
    Name         = "PROD objects",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_s3_bucket_public_access_block" "main_pab" {
  bucket = aws_s3_bucket.main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket" "datalake" {
  bucket = "${var.project}-datalake"

  tags = {
    Name         = "Data lake objects",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_s3_bucket_public_access_block" "datalake_pab" {
  bucket = aws_s3_bucket.datalake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket" "labelstudio" {
  bucket = "${var.project}-labelstudio"

  tags = {
    Name         = "Label Studio objects",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_s3_bucket_public_access_block" "labelstudio_pab" {
  bucket = aws_s3_bucket.labelstudio.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_cors_configuration" "labelstudio_cors" {
  bucket = aws_s3_bucket.labelstudio.bucket

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD", "POST", "PUT"]
    allowed_origins = ["*"]
    expose_headers  = ["x-amz-server-side-encryption", "x-amz-request-id", "x-amz-id-2"]
    max_age_seconds = 3000
  }
}
