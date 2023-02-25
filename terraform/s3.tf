# S3 buckets

resource "aws_s3_bucket" "rtu_dataset_transport_raw" {
  count  = var.s3_enabled ? 1 : 0
  bucket = "rtu-dataset-transport-raw"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  tags = {
    Name         = "RTU Dataset: Transport RAW",
    Organization = "RTU"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_raw_acl" {
  count  = var.s3_enabled ? 1 : 0
  bucket = aws_s3_bucket.rtu_dataset_transport_raw[count.index].id
  acl    = "private"
}

resource "aws_s3_bucket_public_access_block" "rtu_dataset_transport_raw_pab" {
  count  = var.s3_enabled ? 1 : 0
  bucket = aws_s3_bucket.rtu_dataset_transport_raw[count.index].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "rtu_dataset_transport_gen" {
  count  = var.s3_enabled ? 1 : 0
  bucket = "rtu-dataset-transport-gen"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  tags = {
    Name         = "RTU Dataset: Transport generated",
    Organization = "RTU"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_gen_acl" {
  count  = var.s3_enabled ? 1 : 0
  bucket = aws_s3_bucket.rtu_dataset_transport_gen[count.index].id
  acl    = "private"
}

resource "aws_s3_bucket_public_access_block" "rtu_dataset_transport_gen_pab" {
  count  = var.s3_enabled ? 1 : 0
  bucket = aws_s3_bucket.rtu_dataset_transport_gen[count.index].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "rtu_dataset_transport_thumb" {
  count  = var.s3_enabled ? 1 : 0
  bucket = "rtu-dataset-transport-thumb"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  tags = {
    Name         = "RTU Dataset: Transport thumbnails",
    Organization = "RTU"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_thumb_acl" {
  count  = var.s3_enabled ? 1 : 0
  bucket = aws_s3_bucket.rtu_dataset_transport_thumb[count.index].id
  acl    = "public-read"
}
